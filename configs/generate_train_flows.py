from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


VEHICLE = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.111,
    "headwayTime": 2,
}


def load_routes(base_flow: str) -> list[list[str]]:
    with open(base_flow, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes = []
    seen = set()

    for item in data:
        route = item.get("route")
        if not route:
            continue

        key = tuple(route)
        if key not in seen:
            seen.add(key)
            routes.append(route)

    if not routes:
        raise ValueError("No valid routes found in base flow.")

    return routes


def make_gaussian_flow(
    routes: list[list[str]],
    seed: int,
    mean_vehicles: int,
    duration: int,
    sigma_min: int,
    sigma_max: int,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)

    # Random số xe quanh mean
    num_vehicles = int(np.random.normal(mean_vehicles, mean_vehicles * 0.1))
    num_vehicles = max(1, num_vehicles)

    # Random vị trí cao điểm
    peak_center = random.randint(duration // 3, duration * 2 // 3)
    sigma = random.randint(sigma_min, sigma_max)

    flow = []

    for _ in range(num_vehicles):
        start_time = int(np.random.normal(peak_center, sigma))
        start_time = max(0, min(duration - 1, start_time))

        route = random.choice(routes)

        flow.append(
            {
                "vehicle": VEHICLE,
                "route": route,
                "interval": 1.0,
                "startTime": start_time,
                "endTime": start_time,
            }
        )

    flow.sort(key=lambda x: x["startTime"])
    return flow


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-flow",
        default="configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json",
    )
    parser.add_argument("--out-dir", default="configs/train_flows")
    parser.add_argument("--num-files", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--mean-vehicles", type=int, default=8412)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--sigma-min", type=int, default=300)
    parser.add_argument("--sigma-max", type=int, default=700)

    args = parser.parse_args()

    routes = load_routes(args.base_flow)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seed_start, args.seed_start + args.num_files):
        flow = make_gaussian_flow(
            routes=routes,
            seed=seed,
            mean_vehicles=args.mean_vehicles,
            duration=args.duration,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )

        out_path = out_dir / f"gaussian_train_seed_{seed:03d}.json"

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(flow, f, separators=(",", ":"))

        print(f"wrote {out_path} | vehicles={len(flow)}")


if __name__ == "__main__":
    main()