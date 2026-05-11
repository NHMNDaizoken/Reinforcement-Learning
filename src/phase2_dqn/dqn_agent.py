"""Shared DQN components for traffic signal control."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, NamedTuple

import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseLayer(nn.Module):
    """Fully connected layer using standard (out_features, in_features) weight layout."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        fan_in = self.weight.shape[1]
        bound = 1.0 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.t() + self.bias


class QNetwork(nn.Module):
    """MLP mapping one intersection state to action values (128 → 64 → action_dim)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.input_layer = DenseLayer(state_dim, 128)
        self.hidden_layer = DenseLayer(128, 64)
        self.output_layer = DenseLayer(64, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_layer(states))
        hidden = torch.relu(self.hidden_layer(hidden))
        return self.output_layer(hidden)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):  # FIX 1: 5000 → 50000
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more transitions than are in the buffer")

        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss from the piecewise definition."""
    errors = targets - predictions
    absolute_errors = torch.abs(errors)
    quadratic = torch.clamp(absolute_errors, max=delta)
    linear = absolute_errors - quadratic
    losses = 0.5 * quadratic.pow(2) + delta * linear
    return losses.mean()


@dataclass
class SharedDQNAgent:
    state_dim: int
    action_dim: int
    learning_rate: float = 5e-4
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9934  # reaches epsilon_min=0.05 at ~ep 450 of 500
    buffer_capacity: int = 50000
    batch_size: int = 256
    target_update_freq: int = 1000
    huber_delta: float = 1.0
    grad_max_norm: float = 10.0
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.update_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def select_actions(
        self,
        states: list[np.ndarray],
        training: bool = True,
    ) -> list[int]:
        """Select actions for all agents in one batched forward pass."""
        n = len(states)
        explore_mask = [training and random.random() < self.epsilon for _ in range(n)]

        if all(explore_mask):
            return [random.randrange(self.action_dim) for _ in range(n)]

        batch = np.stack([np.asarray(s, dtype=np.float32) for s in states])
        state_tensor = torch.as_tensor(batch, device=self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        greedy = torch.argmax(q_values, dim=1).tolist()

        return [
            random.randrange(self.action_dim) if explore_mask[i] else int(greedy[i])
            for i in range(n)
        ]

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> float:
        """Decay exploration rate once per completed decision episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def update_target_network(self) -> None:
        """Synchronize target-network weights with the online Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self) -> float | None:
        """Sample a minibatch and perform one gradient update. Does NOT decay epsilon."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        state_tensor = to_tensor(states, torch.float32)
        action_tensor = to_tensor(actions, torch.long)
        reward_tensor = to_tensor(rewards, torch.float32)
        next_state_tensor = to_tensor(next_states, torch.float32)
        done_tensor = to_tensor(dones, torch.float32)

        q_values = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor).max(dim=1).values
            targets = reward_tensor + self.gamma * next_q_values * (1.0 - done_tensor)

        loss = huber_loss(q_values, targets, delta=self.huber_delta)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_max_norm)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())
