import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";

type Agent = {
  id: string;
  row: number;
  col: number;
  action: number;
  queue: number;
  phase_pressures: number[];
  selected_pressure: number;
};

type VehicleType = "car" | "bus" | "motorbike";

type Vehicle = {
  id: string;
  x: number;
  y: number;
  angle?: number;
  type?: VehicleType;
  waiting: boolean;
};

type Frame = {
  time: number;
  atl: number;
  throughput: number;
  total_queue: number;
  agents: Agent[];
  vehicles: Vehicle[];
};

type ReplayData = {
  roadnet: string;
  flow: string;
  traffic_level: TrafficLevel;
  algorithm: Algorithm;
  algorithm_label: string;
  available_algorithms: Algorithm[];
  steps: number;
  action_interval: number;
  frames: Frame[];
};

type TrafficLevel = "low" | "medium" | "high";
type Algorithm = "baseline" | "model";

const replays: Record<TrafficLevel, Partial<Record<Algorithm, ReplayData>>> = {
  low: { baseline: require('./data/low.json'), model: undefined },
  medium: { baseline: require('./data/medium.json'), model: undefined },
  high: { baseline: require('./data/high.json'), model: require("./data/high_model.json"), },
};

const trafficOptions: { key: TrafficLevel; label: string; detail: string }[] = [
  { key: "low", label: "Low", detail: "300 veh/h" },
  { key: "medium", label: "Medium", detail: "600 veh/h" },
  { key: "high", label: "High", detail: "900 veh/h" },
];

const algorithmOptions: { key: Algorithm; label: string }[] = [
  { key: "baseline", label: "Baseline" },
  { key: "model", label: "Trained Model" },
];

// FIX: emoji cho từng loại xe
const VEHICLE_EMOJI: Record<VehicleType, string> = {
  car: "🚗",
  bus: "🚌",
  motorbike: "🛵",
};

// FIX: màu nền cho từng loại xe khi đang chờ
const VEHICLE_WAITING_COLOR: Record<VehicleType, string> = {
  car: "#ef4444",
  bus: "#dc2626",
  motorbike: "#f97316",
};

const worldNodes = [170, 410, 650];

export default function App() {
  const { width, height } = useWindowDimensions();
  const compact = width < 820;
  const [traffic, setTraffic] = useState<TrafficLevel>("high");
  const [algorithm, setAlgorithm] = useState<Algorithm>("baseline");
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const replay = replays[traffic][algorithm] ?? replays[traffic].baseline;
  const frame = replay?.frames[frameIndex] ?? replay?.frames[0];
  const unavailable = !replays[traffic][algorithm];

  const focusAgent = useMemo(() => {
    if (!frame) return undefined;
    return [...frame.agents].sort((a, b) => b.queue - a.queue)[0];
  }, [frame]);

  // thống kê xe theo loại
  const vehicleStats = useMemo(() => {
    if (!frame) return { car: 0, bus: 0, motorbike: 0, waiting: 0 };
    const stats = { car: 0, bus: 0, motorbike: 0, waiting: 0 };
    frame.vehicles.forEach((v) => {
      const type = v.type ?? "car";
      stats[type] = (stats[type] ?? 0) + 1;
      if (v.waiting) stats.waiting++;
    });
    return stats;
  }, [frame]);

  useEffect(() => {
    setFrameIndex(0);
  }, [traffic, algorithm]);

  useEffect(() => {
    if (!playing || !replay) return;
    intervalRef.current = setInterval(() => {
      setFrameIndex((current) => (current + 1) % replay.frames.length);
    }, Math.max(45, 280 / speed));
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, replay, speed]);

  if (!replay || !frame) {
    return (
      <View style={styles.emptyScreen}>
        <Text style={styles.emptyTitle}>No replay data</Text>
      </View>
    );
  }

  const mapSize = compact ? Math.min(width, height * 0.72) : Math.min(width - 330, height);

  return (
    <View style={[styles.app, compact && styles.appCompact]}>
      <ScrollView style={[styles.sidebar, compact && styles.sidebarCompact]}>
        <Text style={styles.title}>🚦 BTCK Traffic Dashboard</Text>

        {/* Stats */}
        <View style={styles.statGrid}>
          <Stat label="Vehicles" value={String(frame.vehicles.length)} />
          <Stat label="Step" value={`${Math.round(frame.time)} / ${replay.steps}`} />
          <Stat label="ATL" value={`${frame.atl.toFixed(2)}s`} />
          <Stat label="Throughput" value={String(frame.throughput)} />
          <Stat label="Queue" value={String(frame.total_queue)} />
        </View>

        {/* FIX: thống kê xe theo loại */}
        <Section title="Vehicle Types">
          <View style={styles.vehicleStats}>
            <VehicleStat emoji="🚗" label="Cars" count={vehicleStats.car} />
            <VehicleStat emoji="🚌" label="Buses" count={vehicleStats.bus} />
            <VehicleStat emoji="🛵" label="Motorbikes" count={vehicleStats.motorbike} />
            <VehicleStat emoji="🔴" label="Waiting" count={vehicleStats.waiting} />
          </View>
        </Section>

        <Section title="Traffic Config">
          <Segmented
            items={trafficOptions}
            value={traffic}
            onChange={(value) => setTraffic(value as TrafficLevel)}
          />
        </Section>

        <Section title="Algorithm">
          <Segmented
            items={algorithmOptions}
            value={algorithm}
            onChange={(value) => setAlgorithm(value as Algorithm)}
          />
          {unavailable ? (
            <Text style={styles.warning}>
              Chưa có replay cho model đã train. Sau khi evaluate model, thêm
              file JSON vào web/data để selector này chạy thật.
            </Text>
          ) : null}
        </Section>

        <Section title="Algorithm Details">
          {algorithm === "baseline" ? (
            <View style={styles.infoBox}>
              <Text style={styles.bodyText}>
                MaxPressure Actuated Control không dùng học máy. Mỗi giao lộ
                tính áp lực cho từng pha rồi chọn pha có áp lực lớn nhất.
              </Text>
              <Text style={styles.formula}>pressure(phase) = sum(q_in) − sum(q_out)</Text>
              <Text style={styles.bodyText}>
                Quy trình: đọc lane counts từ CityFlow, tính pressure từng pha,
                chọn argmax, giữ pha trong action interval rồi tính lại.
              </Text>
            </View>
          ) : (
            <View style={styles.infoBox}>
              <Text style={styles.bodyText}>
                Shared DQN dùng một mạng chung cho 9 giao lộ. Action là pha có
                Q-value lớn nhất từ state hiện tại.
              </Text>
              <Text style={styles.formula}>action = argmax_a Q(state, a; θ_shared)</Text>
            </View>
          )}
        </Section>

        <Section title="Busiest Intersection">
          <View style={styles.infoBox}>
            <Text style={styles.bodyStrong}>{focusAgent?.id}</Text>
            <Text style={styles.bodyText}>Queue: {focusAgent?.queue} vehicles</Text>
            <Text style={styles.bodyText}>Selected phase: {focusAgent?.action}</Text>
            <Text style={styles.formula}>
              max({focusAgent?.phase_pressures.join(", ")}) = {focusAgent?.selected_pressure}
            </Text>
            {focusAgent?.phase_pressures.map((pressure, phase) => (
              <Text key={phase} style={styles.bodyText}>
                phase {phase}: pressure = {pressure}
              </Text>
            ))}
          </View>
        </Section>

        <Section title="Controls">
          <View style={styles.buttonRow}>
            <Button label={playing ? "⏸ Pause" : "▶ Play"} onPress={() => setPlaying(!playing)} />
            <Button
              label="⏭ Step"
              secondary
              onPress={() => setFrameIndex((frameIndex + 1) % replay.frames.length)}
            />
          </View>
          <View style={styles.buttonRow}>
            {[0.5, 1, 2, 4].map((value) => (
              <Button
                key={value}
                label={`${value}x`}
                secondary={speed !== value}
                onPress={() => setSpeed(value)}
              />
            ))}
          </View>
        </Section>
      </ScrollView>

      <View style={styles.stage}>
        <TrafficMap frame={frame} size={mapSize} />
      </View>
    </View>
  );
}

function TrafficMap({ frame, size }: { frame: Frame; size: number }) {
  const scale = size / 820;
  const toPx = (value: number) => value * scale;
  const roadWidth = 58;
  const laneWidth = roadWidth / 2;

  return (
    <View style={[styles.map, { width: size, height: size }]}>
      {/* Roads ngang — FIX: vẽ 2 làn rõ hơn */}
      {worldNodes.map((y) => (
        <View key={`h-${y}`}>
          {/* đường ngang */}
          <View style={[styles.road, {
            left: -40 * scale,
            top: (y - roadWidth / 2) * scale,
            width: 900 * scale,
            height: roadWidth * scale,
          }]}>
            {/* vạch giữa */}
            <View style={[styles.centerLineDash, { top: (roadWidth / 2 - 0.5) * scale, height: Math.max(1, scale) }]} />
            {/* vạch làn trái */}
            <View style={[styles.laneMarker, { top: (roadWidth / 4) * scale, height: Math.max(1, 0.5 * scale) }]} />
            {/* vạch làn phải */}
            <View style={[styles.laneMarker, { top: (3 * roadWidth / 4) * scale, height: Math.max(1, 0.5 * scale) }]} />
          </View>
        </View>
      ))}

      {/* Roads dọc — FIX: vẽ 2 làn rõ hơn */}
      {worldNodes.map((x) => (
        <View key={`v-${x}`}>
          <View style={[styles.road, {
            left: (x - roadWidth / 2) * scale,
            top: -40 * scale,
            width: roadWidth * scale,
            height: 900 * scale,
          }]}>
            {/* vạch giữa */}
            <View style={[styles.centerLineDash, {
              left: (roadWidth / 2 - 0.5) * scale,
              width: Math.max(1, scale),
              top: 0,
              bottom: 0,
              height: undefined,
            }]} />
          </View>
        </View>
      ))}

      {/* FIX: Vehicles với emoji theo loại xe */}
      {frame.vehicles.map((vehicle) => {
        const type = vehicle.type ?? "car";
        const emoji = VEHICLE_EMOJI[type];
        const fontSize = type === "bus" ? 14 * scale : type === "motorbike" ? 9 * scale : 11 * scale;
        return (
          <View
            key={vehicle.id}
            style={[
              styles.vehicleContainer,
              vehicle.waiting && { opacity: 0.75 },
              {
                left: toPx(vehicle.x) - 10 * scale,
                top: toPx(vehicle.y) - 10 * scale,
                width: 20 * scale,
                height: 20 * scale,
                transform: vehicle.angle !== undefined
                  ? [{ rotate: `${vehicle.angle}deg` }]
                  : [],
              },
            ]}
          >
            <Text style={{ fontSize: Math.max(8, fontSize), lineHeight: Math.max(8, fontSize) + 2 }}>
              {emoji}
            </Text>
          </View>
        );
      })}

      {/* Intersections */}
      {frame.agents.map((agent) => {
        const x = worldNodes[agent.col];
        const y = worldNodes[agent.row];
        return (
          <View
            key={agent.id}
            style={[
              styles.intersection,
              {
                left: toPx(x) - 40 * scale,
                top: toPx(y) - 40 * scale,
                width: 80 * scale,
                height: 80 * scale,
                borderRadius: 40 * scale,
              },
            ]}
          >
            {/* tín hiệu ngang */}
            <View style={[
              styles.signalHorizontal,
              agent.action === 0 ? styles.signalGo : styles.signalStop,
            ]} />
            {/* tín hiệu dọc */}
            <View style={[
              styles.signalVertical,
              agent.action === 1 ? styles.signalGo : styles.signalStop,
            ]} />
            {/* số xe đang chờ */}
            <Text style={[styles.queueText, { fontSize: Math.max(10, 15 * scale) }]}>
              {agent.queue}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

// Component thống kê xe theo loại
function VehicleStat({ emoji, label, count }: { emoji: string; label: string; count: number }) {
  return (
    <View style={styles.vehicleStatItem}>
      <Text style={styles.vehicleStatEmoji}>{emoji}</Text>
      <Text style={styles.vehicleStatCount}>{count}</Text>
      <Text style={styles.vehicleStatLabel}>{label}</Text>
    </View>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

function Segmented({
  items,
  value,
  onChange,
}: {
  items: { key: string; label: string; detail?: string }[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <View style={styles.segmented}>
      {items.map((item) => (
        <Pressable
          key={item.key}
          style={[styles.segment, value === item.key && styles.segmentActive]}
          onPress={() => onChange(item.key)}
        >
          <Text style={[styles.segmentLabel, value === item.key && styles.segmentLabelActive]}>
            {item.label}
          </Text>
          {item.detail ? <Text style={styles.segmentDetail}>{item.detail}</Text> : null}
        </Pressable>
      ))}
    </View>
  );
}

function Button({
  label,
  onPress,
  secondary,
}: {
  label: string;
  onPress: () => void;
  secondary?: boolean;
}) {
  return (
    <Pressable style={[styles.button, secondary && styles.buttonSecondary]} onPress={onPress}>
      <Text style={[styles.buttonText, secondary && styles.buttonSecondaryText]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  app: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "#1a1f2e",
  },
  appCompact: {
    flexDirection: "column",
  },
  sidebar: {
    width: 320,
    backgroundColor: "#0f1623",
    borderRightWidth: 1,
    borderRightColor: "#1e2d40",
    padding: 16,
  },
  sidebarCompact: {
    width: "100%",
    maxHeight: 390,
    borderRightWidth: 0,
    borderBottomWidth: 1,
    borderBottomColor: "#1e2d40",
  },
  title: {
    color: "#e2e8f0",
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 14,
  },
  statGrid: {
    gap: 2,
  },
  stat: {
    minHeight: 30,
    borderBottomWidth: 1,
    borderBottomColor: "#1e2d40",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  statLabel: {
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: "600",
  },
  statValue: {
    color: "#38bdf8",
    fontSize: 12,
    fontVariant: ["tabular-nums"],
    fontWeight: "700",
  },
  section: {
    marginTop: 16,
  },
  sectionTitle: {
    color: "#64748b",
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  // FIX: vehicle stats
  vehicleStats: {
    flexDirection: "row",
    justifyContent: "space-between",
    backgroundColor: "#1e2d40",
    borderRadius: 8,
    padding: 10,
  },
  vehicleStatItem: {
    alignItems: "center",
    gap: 2,
  },
  vehicleStatEmoji: {
    fontSize: 18,
  },
  vehicleStatCount: {
    color: "#e2e8f0",
    fontSize: 14,
    fontWeight: "700",
  },
  vehicleStatLabel: {
    color: "#64748b",
    fontSize: 10,
  },
  segmented: {
    gap: 6,
  },
  segment: {
    minHeight: 42,
    borderWidth: 1,
    borderColor: "#1e2d40",
    backgroundColor: "#0f1623",
    paddingHorizontal: 10,
    paddingVertical: 8,
    justifyContent: "center",
    borderRadius: 6,
  },
  segmentActive: {
    borderColor: "#38bdf8",
    backgroundColor: "#0c2340",
  },
  segmentLabel: {
    color: "#94a3b8",
    fontSize: 13,
    fontWeight: "700",
  },
  segmentLabelActive: {
    color: "#38bdf8",
  },
  segmentDetail: {
    color: "#475569",
    fontSize: 11,
    marginTop: 2,
  },
  warning: {
    color: "#f59e0b",
    fontSize: 12,
    lineHeight: 17,
    marginTop: 8,
  },
  infoBox: {
    borderWidth: 1,
    borderColor: "#1e2d40",
    backgroundColor: "#0a1628",
    padding: 10,
    borderRadius: 6,
  },
  bodyText: {
    color: "#94a3b8",
    fontSize: 12,
    lineHeight: 18,
  },
  bodyStrong: {
    color: "#e2e8f0",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 4,
  },
  formula: {
    color: "#38bdf8",
    backgroundColor: "#0c2340",
    fontFamily: "monospace",
    fontSize: 12,
    lineHeight: 17,
    marginVertical: 7,
    padding: 7,
    borderRadius: 4,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8,
  },
  button: {
    flex: 1,
    minHeight: 38,
    backgroundColor: "#0284c7",
    borderWidth: 1,
    borderColor: "#0369a1",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 6,
  },
  buttonSecondary: {
    backgroundColor: "#1e2d40",
    borderColor: "#334155",
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "700",
  },
  buttonSecondaryText: {
    color: "#94a3b8",
  },
  stage: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    backgroundColor: "#1a1f2e",
  },
  map: {
    position: "relative",
    overflow: "hidden",
    backgroundColor: "#2d4a3e",  // màu nền xanh lá như vỉa hè
  },
  road: {
    position: "absolute",
    backgroundColor: "#3d4f5c",  // màu đường xám xanh
  },
  centerLineDash: {
    position: "absolute",
    left: 0,
    right: 0,
    backgroundColor: "rgba(255, 220, 80, 0.6)",  // vạch vàng giữa đường
  },
  laneMarker: {
    position: "absolute",
    left: 0,
    right: 0,
    backgroundColor: "rgba(255, 255, 255, 0.15)",  // vạch trắng mờ chia làn
  },
  // FIX: vehicle container thay vì hình chữ nhật
  vehicleContainer: {
    position: "absolute",
    alignItems: "center",
    justifyContent: "center",
  },
  intersection: {
    position: "absolute",
    backgroundColor: "#4a5568",
    alignItems: "center",
    justifyContent: "center",
  },
  signalHorizontal: {
    position: "absolute",
    left: "13%",
    right: "13%",
    top: "45%",
    height: "10%",
    borderRadius: 2,
  },
  signalVertical: {
    position: "absolute",
    top: "13%",
    bottom: "13%",
    left: "45%",
    width: "10%",
    borderRadius: 2,
  },
  signalGo: {
    backgroundColor: "#4ade80",
    shadowColor: "#4ade80",
    shadowOpacity: 0.8,
    shadowRadius: 4,
  },
  signalStop: {
    backgroundColor: "#f43f5e",
    shadowColor: "#f43f5e",
    shadowOpacity: 0.6,
    shadowRadius: 3,
  },
  queueText: {
    color: "#e2e8f0",
    fontWeight: "700",
  },
  emptyScreen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0f1623",
  },
  emptyTitle: {
    color: "#e2e8f0",
    fontSize: 18,
    fontWeight: "700",
  },
});
