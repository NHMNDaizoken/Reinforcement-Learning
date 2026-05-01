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

type Vehicle = {
  id: string;
  x: number;
  y: number;
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
  low: { baseline: require("./data/low.json") },
  medium: { baseline: require("./data/medium.json") },
  high: { baseline: require("./data/high.json") },
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
        <Text style={styles.title}>BTCK Traffic Dashboard</Text>

        <View style={styles.statGrid}>
          <Stat label="Vehicles" value={String(frame.vehicles.length)} />
          <Stat label="Step" value={`${Math.round(frame.time)} / ${replay.steps}`} />
          <Stat label="ATL" value={frame.atl.toFixed(2)} />
          <Stat label="Throughput" value={String(frame.throughput)} />
          <Stat label="Queue" value={String(frame.total_queue)} />
        </View>

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

        <Section title="Baseline Details">
          {algorithm === "baseline" ? (
            <View style={styles.infoBox}>
              <Text style={styles.bodyText}>
                MaxPressure Actuated Control không dùng học máy. Mỗi giao lộ
                tính áp lực cho từng pha rồi chọn pha có áp lực lớn nhất.
              </Text>
              <Text style={styles.formula}>pressure(phase) = sum(q_in) - sum(q_out)</Text>
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
              <Text style={styles.formula}>action = argmax_a Q(state, a; theta_shared)</Text>
            </View>
          )}
        </Section>

        <Section title="Decision">
          <View style={styles.infoBox}>
            <Text style={styles.bodyStrong}>{focusAgent?.id}</Text>
            <Text style={styles.bodyText}>Queue: {focusAgent?.queue}</Text>
            <Text style={styles.bodyText}>Selected phase: {focusAgent?.action}</Text>
            <Text style={styles.formula}>
              max({focusAgent?.phase_pressures.join(", ")}) ={" "}
              {focusAgent?.selected_pressure}
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
            <Button label={playing ? "Pause" : "Play"} onPress={() => setPlaying(!playing)} />
            <Button
              label="Step"
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

  return (
    <View style={[styles.map, { width: size, height: size }]}>
      {worldNodes.map((y) => (
        <Road key={`h-${y}`} horizontal x={-40} y={y} length={900} scale={scale} />
      ))}
      {worldNodes.map((x) => (
        <Road key={`v-${x}`} x={x} y={-40} length={900} scale={scale} />
      ))}

      {frame.vehicles.map((vehicle) => (
        <View
          key={vehicle.id}
          style={[
            styles.vehicle,
            vehicle.waiting && styles.vehicleWaiting,
            {
              left: toPx(vehicle.x) - 7 * scale,
              top: toPx(vehicle.y) - 3 * scale,
              width: 14 * scale,
              height: 6 * scale,
            },
          ]}
        />
      ))}

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
            <View
              style={[
                styles.signalHorizontal,
                agent.action === 0 ? styles.signalGo : styles.signalStop,
              ]}
            />
            <View
              style={[
                styles.signalVertical,
                agent.action === 1 ? styles.signalGo : styles.signalStop,
              ]}
            />
            <Text style={[styles.queueText, { fontSize: Math.max(10, 15 * scale) }]}>
              {agent.queue}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function Road({
  horizontal,
  x,
  y,
  length,
  scale,
}: {
  horizontal?: boolean;
  x: number;
  y: number;
  length: number;
  scale: number;
}) {
  return (
    <View
      style={[
        styles.road,
        horizontal
          ? {
              left: x * scale,
              top: y * scale - 29 * scale,
              width: length * scale,
              height: 58 * scale,
            }
          : {
              left: x * scale - 29 * scale,
              top: y * scale,
              width: 58 * scale,
              height: length * scale,
            },
      ]}
    >
      <View
        style={[
          styles.centerLine,
          horizontal
            ? { left: 0, right: 0, top: 28 * scale, height: Math.max(1, scale) }
            : { top: 0, bottom: 0, left: 28 * scale, width: Math.max(1, scale) },
        ]}
      />
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
    backgroundColor: "#9aa4a3",
  },
  appCompact: {
    flexDirection: "column",
  },
  sidebar: {
    width: 320,
    backgroundColor: "#f8fafc",
    borderRightWidth: 1,
    borderRightColor: "#d5dde5",
    padding: 16,
  },
  sidebarCompact: {
    width: "100%",
    maxHeight: 390,
    borderRightWidth: 0,
    borderBottomWidth: 1,
    borderBottomColor: "#d5dde5",
  },
  title: {
    color: "#111827",
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 14,
  },
  statGrid: {
    gap: 2,
  },
  stat: {
    minHeight: 30,
    borderBottomWidth: 1,
    borderBottomColor: "#e5eaf0",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  statLabel: {
    color: "#111827",
    fontSize: 12,
    fontWeight: "700",
  },
  statValue: {
    color: "#475569",
    fontSize: 12,
    fontVariant: ["tabular-nums"],
  },
  section: {
    marginTop: 16,
  },
  sectionTitle: {
    color: "#334155",
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 8,
  },
  segmented: {
    gap: 8,
  },
  segment: {
    minHeight: 42,
    borderWidth: 1,
    borderColor: "#d5dde5",
    backgroundColor: "#ffffff",
    paddingHorizontal: 10,
    paddingVertical: 8,
    justifyContent: "center",
  },
  segmentActive: {
    borderColor: "#0f7cf4",
    backgroundColor: "#eaf3ff",
  },
  segmentLabel: {
    color: "#334155",
    fontSize: 13,
    fontWeight: "700",
  },
  segmentLabelActive: {
    color: "#075fb8",
  },
  segmentDetail: {
    color: "#64748b",
    fontSize: 11,
    marginTop: 2,
  },
  warning: {
    color: "#b45309",
    fontSize: 12,
    lineHeight: 17,
    marginTop: 8,
  },
  infoBox: {
    borderWidth: 1,
    borderColor: "#dbe3ea",
    backgroundColor: "#ffffff",
    padding: 10,
  },
  bodyText: {
    color: "#475569",
    fontSize: 12,
    lineHeight: 17,
  },
  bodyStrong: {
    color: "#111827",
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 4,
  },
  formula: {
    color: "#111827",
    backgroundColor: "#f1f5f9",
    fontFamily: "monospace",
    fontSize: 12,
    lineHeight: 17,
    marginVertical: 7,
    padding: 7,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8,
  },
  button: {
    flex: 1,
    minHeight: 38,
    backgroundColor: "#0f7cf4",
    borderWidth: 1,
    borderColor: "#0f70e6",
    alignItems: "center",
    justifyContent: "center",
  },
  buttonSecondary: {
    backgroundColor: "#ffffff",
    borderColor: "#cbd5e1",
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "700",
  },
  buttonSecondaryText: {
    color: "#334155",
  },
  stage: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  map: {
    position: "relative",
    overflow: "hidden",
    backgroundColor: "#9aa4a3",
  },
  road: {
    position: "absolute",
    backgroundColor: "#5d7077",
  },
  centerLine: {
    position: "absolute",
    backgroundColor: "rgba(238, 242, 247, 0.9)",
  },
  vehicle: {
    position: "absolute",
    backgroundColor: "#f4d35e",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.25)",
  },
  vehicleWaiting: {
    backgroundColor: "#f97316",
  },
  intersection: {
    position: "absolute",
    backgroundColor: "#6e7c78",
    alignItems: "center",
    justifyContent: "center",
  },
  signalHorizontal: {
    position: "absolute",
    left: "13%",
    right: "13%",
    top: "45%",
    height: "10%",
  },
  signalVertical: {
    position: "absolute",
    top: "13%",
    bottom: "13%",
    left: "45%",
    width: "10%",
  },
  signalGo: {
    backgroundColor: "#84ff00",
  },
  signalStop: {
    backgroundColor: "#ff2f66",
  },
  queueText: {
    color: "#ffffff",
    fontWeight: "700",
  },
  emptyScreen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyTitle: {
    color: "#111827",
    fontSize: 18,
    fontWeight: "700",
  },
});
