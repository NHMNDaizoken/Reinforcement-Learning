import json
import os
import random
from pathlib import Path

random.seed(42)

VEHICLE_PARAMS = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.111,
    "headwayTime": 2.0
}

def extract_routes_from_dataset(dataset_flow_path):
    with open(dataset_flow_path, 'r') as f:
        original_flows = json.load(f)
    routes = []
    seen = set()
    for flow in original_flows:
        route_tuple = tuple(flow['route'])
        if route_tuple not in seen:
            seen.add(route_tuple)
            routes.append(list(route_tuple))
    return routes

def create_vehicle(route, spawn_time):
    return {
        "vehicle": VEHICLE_PARAMS,
        "route": route,
        "interval": 1.0,
        "startTime": int(spawn_time),
        "endTime": int(spawn_time)
    }

def generate_flat_flow(routes, total_vehicles, filename):
    flows = []
    MAX_SPAWN_TIME = 2700.0 
    interval = MAX_SPAWN_TIME / total_vehicles
    
    for i in range(total_vehicles):
        route = random.choice(routes)
        spawn_time = i * interval
        flows.append(create_vehicle(route, spawn_time))

    flows.sort(key=lambda x: x["startTime"])
    with open(filename, 'w') as f:
        json.dump(flows, f, indent=2)
    print(f"Created flat flow ({total_vehicles} vehicles in first 2700s): {filename}")

def generate_peak_flow(routes, total_vehicles, filename):
    flows = []
    v1 = int(total_vehicles * 0.15)
    v2 = int(total_vehicles * 0.70)
    v3 = total_vehicles - v1 - v2

    # Phase 1: 0 - 900s
    int1 = 900.0 / max(1, v1)
    for i in range(v1):
        flows.append(create_vehicle(random.choice(routes), i * int1))

    # Phase 2: 900 - 1800s
    int2 = 900.0 / max(1, v2)
    for i in range(v2):
        flows.append(create_vehicle(random.choice(routes), 900 + (i * int2)))

    # Phase 3: 1800 - 2700s
    int3 = 900.0 / max(1, v3) 
    for i in range(v3):
        flows.append(create_vehicle(random.choice(routes), 1800 + (i * int3)))

    flows.sort(key=lambda x: x["startTime"])
    with open(filename, 'w') as f:
        json.dump(flows, f, indent=2)
    print(f"Created peak flow ({total_vehicles} vehicles, peak at 900s-1800s): {filename}")

if __name__ == "__main__":
    DATASET_FLOW_PATH = "configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json"
    if not Path(DATASET_FLOW_PATH).exists():
        print(f"Missing dataset flow file: {DATASET_FLOW_PATH}")
        exit(1)

    routes = extract_routes_from_dataset(DATASET_FLOW_PATH)
    os.makedirs("configs", exist_ok=True)

    legacy_levels = {
        "low": 300,
        "medium": 600,
        "high": 900,
    }
    for label, total in legacy_levels.items():
        generate_flat_flow(routes, total, f"configs/flow_{label}_flat.json")
        generate_peak_flow(routes, total, f"configs/flow_{label}_peak.json")

    smooth_levels = [300, 900, 1800, 3600, 6000]
    for total in smooth_levels:
        generate_flat_flow(routes, total, f"configs/flow_demand_{total}_flat.json")
        generate_peak_flow(routes, total, f"configs/flow_demand_{total}_peak.json")
    
    print("\nGenerated demand curriculum flows with 900s of clearance headroom.")
