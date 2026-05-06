from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


BASE_FLOW = (
    "configs/syn_3x3_gaussian_500_1h/"
    "syn_3x3_gaussian_500_1h.json"
)

OUT_DIR = Path("configs/eval_flows")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_routes(base_flow_path: str) -> list[list[str]]:
    with open(base_flow_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    routes = []
    seen = set()

    for item in base:
        route = item.get("route")

        if not route:
            continue

        key = tuple(route)

        if key not in seen:
            seen.add(key)
            routes.append(route)

    return routes


ROUTES = load_routes(BASE_FLOW)


def make_flat_flow(
    seed: int,
    num_vehicles: int,
    duration: int = 3600,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)

    flow = []

    interval = duration / num_vehicles

    for i in range(num_vehicles):
        start_time = int(i * interval)
        start_time = max(0, min(duration - 1, start_time))

        flow.append(
            {
                "vehicle": VEHICLE,
                "route": random.choice(ROUTES),
                "interval": 1.0,
                "startTime": start_time,
                "endTime": start_time,
            }
        )

    return flow


def make_peak_flow(
    seed: int,
    num_vehicles: int,
    duration: int = 3600,
    sigma: int = 450,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)

    peak_center = duration // 2

    flow = []

    for _ in range(num_vehicles):
        start_time = int(np.random.normal(peak_center, sigma))
        start_time = max(0, min(duration - 1, start_time))

        flow.append(
            {
                "vehicle": VEHICLE,
                "route": random.choice(ROUTES),
                "interval": 1.0,
                "startTime": start_time,
                "endTime": start_time,
            }
        )

    flow.sort(key=lambda x: x["startTime"])
    return flow


BENCHMARKS = [
    ("low_flat_eval.json", "flat", 900, 101),
    ("medium_flat_eval.json", "flat", 3000, 102),
    ("high_flat_eval.json", "flat", 6000, 103),

    ("low_peak_eval.json", "peak", 900, 201),
    ("medium_peak_eval.json", "peak", 3000, 202),
    ("high_peak_eval.json", "peak", 6000, 203),
]


def save_flow(path: Path, flow: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(flow, f, separators=(",", ":"))


def main() -> None:
    for filename, flow_type, vehicles, seed in BENCHMARKS:

        if flow_type == "flat":
            flow = make_flat_flow(
                seed=seed,
                num_vehicles=vehicles,
            )

        else:
            flow = make_peak_flow(
                seed=seed,
                num_vehicles=vehicles,
            )

        out_path = OUT_DIR / filename

        save_flow(out_path, flow)

        print(
            f"wrote {out_path} | "
            f"type={flow_type} | "
            f"vehicles={vehicles}"
        )


if __name__ == "__main__":
    main()