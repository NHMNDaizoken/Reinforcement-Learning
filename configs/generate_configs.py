import json
import os
import random
from pathlib import Path

# Fix seed để kết quả tái tạo được
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
    interval = 3600.0 / total_vehicles
    for i in range(total_vehicles):
        route = random.choice(routes) # Lấy ngẫu nhiên tuyến đường
        spawn_time = i * interval
        flows.append(create_vehicle(route, spawn_time))

    flows.sort(key=lambda x: x["startTime"])
    with open(filename, 'w') as f:
        json.dump(flows, f, indent=2)
    print(f"✅ Đã tạo Flat Flow ({total_vehicles} xe/h): {filename}")

def generate_peak_flow(routes, total_vehicles, filename):
    flows = []
    v1 = int(total_vehicles * 0.15)
    v2 = int(total_vehicles * 0.70)
    v3 = total_vehicles - v1 - v2

    # Phase 1: 0 - 900s
    int1 = 900.0 / max(1, v1)
    for i in range(v1):
        flows.append(create_vehicle(random.choice(routes), i * int1))

    # Phase 2: 900 - 1800s (Cao điểm)
    int2 = 900.0 / max(1, v2)
    for i in range(v2):
        flows.append(create_vehicle(random.choice(routes), 900 + (i * int2)))

    # Phase 3: 1800 - 3600s
    int3 = 1800.0 / max(1, v3)
    for i in range(v3):
        flows.append(create_vehicle(random.choice(routes), 1800 + (i * int3)))

    flows.sort(key=lambda x: x["startTime"])
    with open(filename, 'w') as f:
        json.dump(flows, f, indent=2)
    print(f"🔥 Đã tạo Peak Flow ({total_vehicles} xe/h): {filename}")

if __name__ == "__main__":
    DATASET_FLOW_PATH = "configs/syn_3x3_gaussian_500_1h/syn_3x3_gaussian_500_1h.json"
    if not Path(DATASET_FLOW_PATH).exists():
        print(f"❌ Lỗi: Không tìm thấy file")
        exit(1)

    routes = extract_routes_from_dataset(DATASET_FLOW_PATH)
    os.makedirs("configs", exist_ok=True)

    # Khởi tạo data
    generate_flat_flow(routes, 300, "configs/flow_low_flat.json")
    generate_flat_flow(routes, 600, "configs/flow_medium_flat.json")
    generate_flat_flow(routes, 900, "configs/flow_high_flat.json")

    generate_peak_flow(routes, 300, "configs/flow_low_peak.json")
    generate_peak_flow(routes, 600, "configs/flow_medium_peak.json")
    generate_peak_flow(routes, 900, "configs/flow_high_peak.json")
    
    print("\n🎉 Đã fix lỗi kẹt xe! Xe đã được rải đều mượt mà.")