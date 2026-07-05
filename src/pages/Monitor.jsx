import { useEffect, useRef, useState } from "react";
import {
  Camera,
  Cctv,
  Cpu,
  Gauge,
  Info,
  Navigation,
  Play,
  Radio,
  ScanEye,
  Square,
  TriangleAlert,
  Wifi,
  WifiOff,
} from "lucide-react";
import PageHeader from "../components/PageHeader.jsx";

const DEFAULT_ENDPOINT = "ws://roboss-robot.local:8765";
const MAX_LOG = 40;

const ACTION_SCRIPT = [
  { kind: "move", text: "Navigating to aisle 4, rack B" },
  { kind: "detect", text: "Detected unstable_pallet (0.94)" },
  { kind: "capture", text: "Captured inspection frame #{n}" },
  { kind: "hazard", text: "Hazard flagged: leaning boxes on lower rack" },
  { kind: "detect", text: "Detected shrink_wrap tear (0.81)" },
  { kind: "move", text: "Adjusting camera angle to high-angle view" },
  { kind: "detect", text: "Tracking forklift in lane 2 (0.88)" },
  { kind: "capture", text: "Captured inspection frame #{n}" },
  { kind: "info", text: "Uploading annotated clip to dataset queue" },
  { kind: "move", text: "Advancing 1.2m along floor marking" },
  { kind: "hazard", text: "Hazard flagged: fluid spill near pallet" },
  { kind: "detect", text: "Detected damaged_box (0.77)" },
];

const LOG_ICONS = {
  move: Navigation,
  detect: ScanEye,
  capture: Camera,
  hazard: TriangleAlert,
  info: Info,
};

const LOG_TONES = {
  move: "text-sage-300",
  detect: "text-sage-100",
  capture: "text-sage-200",
  hazard: "text-[#ff3b6b]",
  info: "text-sage-300",
};

const BASE_DETECTIONS = [
  { id: "d1", label: "unstable_pallet", conf: 0.94, tone: "hazard", x: 12, y: 46, w: 30, h: 38 },
  { id: "d2", label: "cardboard_box", conf: 0.86, tone: "object", x: 52, y: 30, w: 22, h: 26 },
  { id: "d3", label: "rack", conf: 0.79, tone: "object", x: 70, y: 12, w: 24, h: 70 },
];

function clock() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function jitter(value, amount, min, max) {
  const next = value + (Math.random() - 0.5) * amount;
  return Math.min(max, Math.max(min, next));
}

function TelemetryCard({ icon: Icon, label, value, tone = "text-white" }) {
  return (
    <div className="rounded-lg border border-surface-700 bg-surface-900 p-3 sm:p-4">
      <div className="flex items-center gap-2 text-sage-400">
        <Icon size={15} aria-hidden="true" />
        <span className="text-xs font-medium uppercase tracking-label">{label}</span>
      </div>
      <div className={`mt-1.5 text-xl font-semibold tracking-tight sm:text-2xl ${tone}`}>{value}</div>
    </div>
  );
}

export default function Monitor() {
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [status, setStatus] = useState("disconnected");
  const [log, setLog] = useState([]);
  const [telemetry, setTelemetry] = useState({ fps: 0, latencyMs: 0, detections: 0, uptime: 0 });
  const [detections, setDetections] = useState(BASE_DETECTIONS);
  const timersRef = useRef([]);
  const stepRef = useRef(0);
  const captureRef = useRef(0);

  const connected = status === "connected";

  function clearTimers() {
    timersRef.current.forEach((id) => window.clearInterval(id));
    timersRef.current = [];
  }

  useEffect(() => clearTimers, []);

  function pushLog(entry) {
    setLog((current) => [{ id: `${Date.now()}-${Math.random()}`, time: clock(), ...entry }, ...current].slice(0, MAX_LOG));
  }

  function connect() {
    if (status !== "disconnected") {
      return;
    }
    setStatus("connecting");
    pushLog({ kind: "info", text: `Connecting to ${endpoint}...` });

    const connectTimer = window.setTimeout(() => {
      setStatus("connected");
      setTelemetry({ fps: 27, latencyMs: 62, detections: BASE_DETECTIONS.length, uptime: 0 });
      pushLog({ kind: "info", text: "Robot online. Streaming live feed." });

      const actionTimer = window.setInterval(() => {
        const template = ACTION_SCRIPT[stepRef.current % ACTION_SCRIPT.length];
        stepRef.current += 1;
        if (template.text.includes("#{n}")) {
          captureRef.current += 1;
        }
        pushLog({
          kind: template.kind,
          text: template.text.replace("#{n}", String(captureRef.current)),
        });
      }, 1800);

      const telemetryTimer = window.setInterval(() => {
        setTelemetry((current) => ({
          fps: Math.round(jitter(current.fps || 27, 4, 22, 30)),
          latencyMs: Math.round(jitter(current.latencyMs || 62, 20, 38, 95)),
          detections: 2 + Math.floor(Math.random() * 3),
          uptime: current.uptime + 1,
        }));
        setDetections((current) =>
          current.map((box) => ({
            ...box,
            x: jitter(box.x, 3, 4, 70),
            y: jitter(box.y, 3, 6, 58),
            conf: Math.min(0.99, Math.max(0.6, jitter(box.conf, 0.06, 0.6, 0.99))),
          })),
        );
      }, 1000);

      timersRef.current.push(actionTimer, telemetryTimer);
    }, 1100);

    timersRef.current.push(connectTimer);
  }

  function disconnect() {
    clearTimers();
    stepRef.current = 0;
    setStatus("disconnected");
    setTelemetry({ fps: 0, latencyMs: 0, detections: 0, uptime: 0 });
    setDetections(BASE_DETECTIONS);
    pushLog({ kind: "info", text: "Disconnected from robot." });
  }

  const uptimeLabel = `${Math.floor(telemetry.uptime / 60)
    .toString()
    .padStart(2, "0")}:${(telemetry.uptime % 60).toString().padStart(2, "0")}`;

  return (
    <>
      <PageHeader
        title="Live Monitor"
        subtitle="Connect the robot to watch its live camera feed, real-time detections and action log."
      >
        <div className="flex w-full items-center gap-2 sm:w-auto">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${
              connected
                ? "border border-[#ff3b6b]/40 bg-[#ff3b6b]/10 text-[#ff3b6b]"
                : status === "connecting"
                  ? "border border-accent-500/30 bg-accent-500/5 text-accent-300"
                  : "border border-surface-600 bg-surface-850 text-sage-400"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                connected ? "animate-pulse bg-[#ff3b6b]" : status === "connecting" ? "bg-accent-400" : "bg-sage-500"
              }`}
            />
            {connected ? "LIVE" : status === "connecting" ? "Connecting" : "Offline"}
          </span>
        </div>
      </PageHeader>

      <div className="mb-6 flex flex-col items-stretch gap-3 rounded-lg border border-surface-700 bg-surface-900 p-4 sm:flex-row sm:flex-wrap sm:items-center">
        <div className="flex items-center gap-2 text-sage-400">
          <Radio size={16} aria-hidden="true" />
          <span className="text-xs font-medium uppercase tracking-label">Robot endpoint</span>
        </div>
        <input
          value={endpoint}
          onChange={(event) => setEndpoint(event.target.value)}
          disabled={status !== "disconnected"}
          className="h-10 min-w-0 flex-1 rounded-md border border-surface-600 bg-surface-950 px-3 text-sm text-sage-50 outline-none transition focus:border-sage-400 focus:ring-2 focus:ring-white/10 disabled:opacity-60"
          placeholder="ws://robot-host:port"
        />
        {connected || status === "connecting" ? (
          <button
            type="button"
            onClick={disconnect}
            className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-[#ff3b6b]/40 bg-[#ff3b6b]/10 px-4 text-sm font-medium text-[#ff3b6b] transition hover:bg-[#ff3b6b]/20 sm:w-auto"
          >
            <Square size={15} aria-hidden="true" />
            Disconnect
          </button>
        ) : (
          <button
            type="button"
            onClick={connect}
            className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-4 text-sm font-medium text-[#0b0714] shadow-[0_0_20px_rgba(241,61,245,0.4)] transition hover:brightness-110 sm:w-auto"
          >
            <Play size={15} aria-hidden="true" />
            Connect
          </button>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <div className="flex flex-col gap-4">
          <div className="relative aspect-video overflow-hidden rounded-lg border border-surface-700 bg-black">
            <div
              className="absolute inset-0 opacity-[0.12]"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(161,161,161,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(161,161,161,0.6) 1px, transparent 1px)",
                backgroundSize: "40px 40px",
              }}
            />

            {connected ? (
              <>
                {detections.map((box) => {
                  const isHazard = box.tone === "hazard";
                  const color = isHazard ? "#ff3b6b" : "#3cf28a";
                  return (
                    <div
                      key={box.id}
                      className="absolute rounded-sm transition-all duration-1000 ease-linear"
                      style={{
                        left: `${box.x}%`,
                        top: `${box.y}%`,
                        width: `${box.w}%`,
                        height: `${box.h}%`,
                        border: `2px solid ${color}`,
                        boxShadow: `0 0 12px ${color}55`,
                      }}
                    >
                      <span
                        className="absolute -top-5 left-0 max-w-[150px] truncate whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-medium text-white sm:-top-6 sm:max-w-none sm:text-[11px]"
                        style={{ backgroundColor: color }}
                      >
                        {box.label} {(box.conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}

                <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/60 px-2 py-1 text-[10px] font-medium text-[#ff3b6b] sm:left-3 sm:top-3 sm:text-[11px]">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#ff3b6b]" />
                  LIVE
                </div>
                <div className="absolute right-2 top-2 rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-sage-300 sm:right-3 sm:top-3 sm:text-[11px]">
                  {clock()} - {telemetry.fps} FPS
                </div>
                <div className="absolute bottom-2 left-2 max-w-[calc(100%-1rem)] truncate rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-sage-300 sm:bottom-3 sm:left-3 sm:text-[11px]">
                  CAM-01 - aisle 4 / rack B
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-4 text-center sm:gap-3">
                <Cctv className="text-sage-500" size={28} aria-hidden="true" />
                <p className="text-base font-medium text-white sm:text-lg">
                  {status === "connecting" ? "Establishing link..." : "No signal"}
                </p>
                <p className="max-w-sm text-xs text-sage-400 sm:text-sm">
                  {status === "connecting"
                    ? "Handshaking with the robot stream."
                    : "Connect the robot to start the live feed and detection overlay."}
                </p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2 sm:grid-cols-4 sm:gap-4">
            <TelemetryCard
              icon={connected ? Wifi : WifiOff}
              label="Status"
              value={connected ? "Online" : "Offline"}
              tone={connected ? "text-[#3cf28a]" : "text-sage-400"}
            />
            <TelemetryCard icon={Gauge} label="FPS" value={telemetry.fps} />
            <TelemetryCard icon={Cpu} label="Latency" value={`${telemetry.latencyMs} ms`} />
            <TelemetryCard icon={ScanEye} label="Objects" value={telemetry.detections} />
          </div>
        </div>

        <div className="flex min-h-[320px] flex-col rounded-lg border border-surface-700 bg-surface-900 sm:min-h-[420px]">
          <div className="flex items-center justify-between border-b border-surface-700 px-4 py-3">
            <span className="text-sm font-medium text-white">Action log</span>
            {connected ? <span className="font-mono text-xs text-sage-500">uptime {uptimeLabel}</span> : null}
          </div>
          <div className="flex-1 space-y-1 overflow-y-auto p-3">
            {log.length ? (
              log.map((entry) => {
                const Icon = LOG_ICONS[entry.kind] || Info;
                return (
                  <div key={entry.id} className="flex items-start gap-2.5 rounded-md px-2 py-1.5 hover:bg-surface-850">
                    <Icon size={15} className={`mt-0.5 shrink-0 ${LOG_TONES[entry.kind] || "text-sage-300"}`} aria-hidden="true" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm leading-snug text-sage-100">{entry.text}</p>
                      <span className="font-mono text-[11px] text-sage-500">{entry.time}</span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="flex h-full items-center justify-center px-6 text-center text-sm text-sage-500">
                The robot action log will stream here once connected.
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
