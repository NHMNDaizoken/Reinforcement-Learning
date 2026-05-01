import numpy as np
import torch
from torch import nn

from src.phase2_dqn.dqn_agent import DenseLayer, QNetwork, ReplayBuffer, SharedDQNAgent, huber_loss


def test_dense_layer_matches_manual_affine_math():
    # weight shape is (out_features, in_features): each row is one output neuron's weights
    layer = DenseLayer(in_features=2, out_features=2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5]))

    output = layer(torch.tensor([[2.0, 3.0]], dtype=torch.float32))

    # output[0] = 1*2 + 2*3 + 0.5 = 8.5
    # output[1] = 3*2 + 4*3 - 0.5 = 17.5
    assert torch.allclose(output, torch.tensor([[8.5, 17.5]]))


def test_q_network_forward_output_shape():
    network = QNetwork(state_dim=12, action_dim=4)
    states = torch.zeros((5, 12), dtype=torch.float32)

    q_values = network(states)

    assert q_values.shape == (5, 4)
    assert not any(isinstance(module, nn.Sequential) for module in network.modules())


def test_huber_loss_matches_piecewise_definition():
    predictions = torch.tensor([0.0, 2.0, 5.0])
    targets = torch.tensor([0.5, 0.0, 1.0])

    loss = huber_loss(predictions, targets, delta=1.0)

    assert torch.isclose(loss, torch.tensor((0.125 + 1.5 + 3.5) / 3))


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=5000)
    state_dim = 6

    for index in range(10):
        state = np.full(state_dim, index, dtype=np.float32)
        next_state = state + 1
        buffer.push(state, index % 3, float(index), next_state, index % 2 == 0)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)

    assert states.shape == (4, state_dim)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, state_dim)
    assert dones.shape == (4,)


def test_epsilon_decay_floor():
    agent = SharedDQNAgent(state_dim=4, action_dim=2)

    for _ in range(2000):
        agent.decay_epsilon()

    assert agent.epsilon == agent.epsilon_min


def test_select_actions_returns_one_action_per_state():
    agent = SharedDQNAgent(state_dim=4, action_dim=2, epsilon=0.0)
    states = [
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    ]

    actions = agent.select_actions(states)

    assert len(actions) == len(states)
    assert all(0 <= action < agent.action_dim for action in actions)


def test_update_with_enough_fake_transitions_does_not_crash():
    agent = SharedDQNAgent(state_dim=5, action_dim=3, batch_size=64)

    for index in range(64):
        state = np.full(5, index / 64, dtype=np.float32)
        next_state = state + 0.1
        agent.remember(
            state=state,
            action=index % agent.action_dim,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    loss = agent.update()

    assert isinstance(loss, float)
    assert agent.update_count == 1
