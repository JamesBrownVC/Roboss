import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Radio, WifiOff } from "lucide-react";
import { getLogs, openLogStream } from "../lib/api.js";
import RobotDog from "./RobotDog.jsx";

const MAX_ENTRIES = 500;
const TERMINAL_STATUSES = ["completed", "failed", "partial"];

const STAGES = [
  { key: "queue", label: "Queue" },
  { key: "intent", label: "Intent" },
  { key: "contract", label: "Contract" },
  { key: "scenarios", label: "Scenarios" },
  { key: "prompts", label: "Prompts & anchors" },
  { key: "render", label: "Video render" },
  { key: "verify", label: "Verify & labels" },
  { key: "export", label: "Export" },
];

const STAGE_INDEX = new Map(STAGES.map((stage, index) => [stage.key, index]));

const STATUS_LABELS = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  warning: "Warning",
  failed: "Failed",
};

const STATUS_STYLES = {
  queued: {
    node: "border-surface-600 bg-surface-950 text-sage-500",
    badge: "border-surface-600 bg-surface-950 text-sage-400",
    dot: "bg-surface-600",
    line: "bg-surface-700",
  },
  running: {
    node: "border-neon-cyan/60 bg-neon-cyan/10 text-white shadow-[0_0_18px_rgba(47,232,234,0.18)]",
    badge: "border-neon-cyan/50 bg-neon-cyan/10 text-neon-cyan",
    dot: "bg-neon-cyan shadow-[0_0_10px_rgba(47,232,234,0.7)]",
    line: "bg-neon-cyan",
  },
  done: {
    node: "border-neon-green/40 bg-neon-green/10 text-sage-100",
    badge: "border-neon-green/40 bg-neon-green/10 text-neon-green",
    dot: "bg-neon-green shadow-[0_0_10px_rgba(60,242,138,0.55)]",
    line: "bg-neon-green",
  },
  warning: {
    node: "border-accent-500/45 bg-accent-500/10 text-accent-200",
    badge: "border-accent-500/40 bg-accent-500/10 text-accent-300",
    dot: "bg-accent-400 shadow-[0_0_10px_rgba(255,176,32,0.55)]",
    line: "bg-accent-400",
  },
  failed: {
    node: "border-neon-red/50 bg-neon-red/10 text-neon-red",
    badge: "border-neon-red/45 bg-neon-red/10 text-neon-red",
    dot: "bg-neon-red shadow-[0_0_10px_rgba(255,59,107,0.55)]",
    line: "bg-neon-red",
  },
};

function includesAny(value, fragments) {
  return fragments.some((fragment) => value.includes(fragment));
}

function stageForEntry(entry) {
  const message = String(entry?.message || "").toLowerCase();
  const agent = String(entry?.agent || "").toLowerCase();

  if (!message && !agent) {
    return null;
  }
  if (message.includes("batch") && message.includes("finished")) {
    return "export";
  }
  if (message.includes("pipeline complete")) {
    return "prompts";
  }
  if (message.includes("starting") && message.includes("video job")) {
    return "render";
  }
  if (
    (message.includes("starting agent pipeline") && message.includes("intent")) ||
    message.includes("batch ") ||
    message.includes("uploaded ")
  ) {
    return "queue";
  }
  if (
    agent === "omni" ||
    agent === "veo" ||
    includesAny(message, ["gemini omni", "video output", "saved "])
  ) {
    return "render";
  }
  if (
    agent === "verifier" ||
    includesAny(message, ["verifier", "annotation", "extracted tracks", "labeled preview", "label"])
  ) {
    return "verify";
  }
  if (
    agent === "compiler" ||
    agent === "canvas" ||
    includesAny(message, ["compiling prompts", "keyframes", "canvas", "start frame", "visual anchors", "anchor"])
  ) {
    return "prompts";
  }
  if (
    agent === "scenarios" ||
    agent === "validator" ||
    includesAny(message, ["planning ", "scenario", "repair", "dropping", "violate"])
  ) {
    return "scenarios";
  }
  if (agent === "contract" || message.includes("world contract") || message.includes("contract")) {
    return "contract";
  }
  if (agent === "intent" || message.includes("parsing intent") || message.includes("intent")) {
    return "intent";
  }
  return null;
}

function stageIndexFromBatch(batch) {
  if (!batch) {
    return -1;
  }
  if (TERMINAL_STATUSES.includes(batch.status)) {
    return STAGES.length - 1;
  }
  if (batch.status === "queued") {
    return STAGE_INDEX.get("queue");
  }

  const jobs = Array.isArray(batch.jobs) ? batch.jobs : [];
  if (
    jobs.some((job) =>
      ["reviewing", "labeling", "rendering"].includes(job.status) ||
      ["running", "passed", "failed"].includes(job.reviewStatus) ||
      job.labelStatus === "running" ||
      job.labeledVideoUrl,
    )
  ) {
    return STAGE_INDEX.get("verify");
  }
  if (jobs.some((job) => job.status === "generating" || job.videoUrl)) {
    return STAGE_INDEX.get("render");
  }
  if (batch.status === "running") {
    return STAGE_INDEX.get("queue");
  }
  return -1;
}

function deriveCircuitState(batch, entries) {
  const warningStages = new Set();
  let latestStageIndex = batch ? STAGE_INDEX.get("queue") : -1;
  let failedStageIndex = -1;
  let lastStageKey = batch ? "queue" : null;
  let lastEntry = null;

  for (const entry of entries) {
    const stageKey = stageForEntry(entry);
    if (!stageKey) {
      continue;
    }
    const stageIndex = STAGE_INDEX.get(stageKey);
    latestStageIndex = Math.max(latestStageIndex, stageIndex);
    lastStageKey = stageKey;
    lastEntry = entry;

    if (entry.level === "warn") {
      warningStages.add(stageKey);
    } else if (entry.level === "error") {
      failedStageIndex = stageIndex;
      warningStages.delete(stageKey);
    }
  }

  latestStageIndex = Math.max(latestStageIndex, stageIndexFromBatch(batch));

  const terminal = batch ? TERMINAL_STATUSES.includes(batch.status) : false;
  if (batch?.status === "failed" && failedStageIndex < 0) {
    failedStageIndex = latestStageIndex >= 0 ? latestStageIndex : STAGE_INDEX.get("export");
  }
  if (batch?.status === "partial") {
    warningStages.add("export");
  }

  const activeIndex = terminal && failedStageIndex < 0 ? STAGE_INDEX.get("export") : Math.max(0, latestStageIndex);
  const statuses = STAGES.map((stage, index) => {
    if (!batch) {
      return "queued";
    }
    if (failedStageIndex >= 0) {
      if (index < failedStageIndex) {
        return "done";
      }
      if (index === failedStageIndex) {
        return "failed";
      }
      return "queued";
    }
    if (batch.status === "completed") {
      return "done";
    }
    if (index < activeIndex) {
      return warningStages.has(stage.key) ? "warning" : "done";
    }
    if (index === activeIndex) {
      if (batch.status === "partial" && stage.key === "export") {
        return "warning";
      }
      if (warningStages.has(stage.key)) {
        return "warning";
      }
      return TERMINAL_STATUSES.includes(batch.status) ? "done" : "running";
    }
    return "queued";
  });

  return {
    activeIndex,
    lastEntry,
    lastStage: STAGES[STAGE_INDEX.get(lastStageKey) ?? activeIndex],
    running: batch ? ["queued", "running"].includes(batch.status) && failedStageIndex < 0 : false,
    statuses,
  };
}

function StatusBadge({ status }) {
  const styles = STATUS_STYLES[status] || STATUS_STYLES.queued;
  return (
    <span className={`inline-flex min-h-6 items-center rounded-md border px-2 text-[11px] font-medium ${styles.badge}`}>
      {STATUS_LABELS[status] || status}
    </span>
  );
}

function StageCard({ stage, index, status, active, running, jumpKey }) {
  const styles = STATUS_STYLES[status] || STATUS_STYLES.queued;
  return (
    <div className={`relative min-h-[112px] rounded-lg border p-3 transition ${styles.node}`}>
      {active ? (
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 lg:hidden">
          <div key={jumpKey} className={jumpKey > 0 ? "roboss-dog-jump" : ""}>
            <RobotDog running={running || active} className="h-[42px] w-[70px]" />
          </div>
        </div>
      ) : null}
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-sage-500">{String(index + 1).padStart(2, "0")}</span>
        <span className={`h-2 w-2 rounded-full ${styles.dot}`} />
      </div>
      <p className="min-h-[34px] text-sm font-semibold leading-tight text-inherit">{stage.label}</p>
      <div className="mt-3">
        <StatusBadge status={status} />
      </div>
    </div>
  );
}

function DesktopCircuit({ activeIndex, running, statuses, jumpKey }) {
  const progressRatio = activeIndex / (STAGES.length - 1);
  const progressPercent = progressRatio * 100;
  const dogLeft = `calc(${progressPercent}% + ${3 - progressRatio * 6}rem)`;
  const progressWidth = activeIndex === 0 ? "0px" : `calc(${progressPercent}% + ${1 - progressRatio * 6}rem)`;
  return (
    <div className="relative hidden min-h-[220px] overflow-hidden rounded-lg border border-surface-700 bg-surface-950/65 px-6 pb-6 pt-5 lg:block">
      <div className="relative z-10 grid grid-cols-8 gap-3">
        {STAGES.map((stage, index) => (
          <div key={stage.key} className="min-w-0">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[11px] text-sage-500">{String(index + 1).padStart(2, "0")}</span>
              <span className={`h-2 w-2 rounded-full ${STATUS_STYLES[statuses[index]].dot}`} />
            </div>
            <p className="truncate text-xs font-semibold text-sage-100">{stage.label}</p>
            <div className="mt-2">
              <StatusBadge status={statuses[index]} />
            </div>
          </div>
        ))}
      </div>

      <div className="absolute inset-x-8 bottom-[64px] h-px bg-surface-700" aria-hidden="true" />
      <div
        className="absolute bottom-[63px] left-8 h-[3px] rounded-full bg-gradient-to-r from-neon-magenta via-neon-violet to-neon-cyan transition-all duration-500"
        style={{ width: progressWidth }}
        aria-hidden="true"
      />
      <div
        className="absolute bottom-[72px] z-20 -translate-x-1/2 transition-[left] duration-500 ease-out"
        style={{ left: dogLeft }}
        aria-hidden="true"
      >
        <div key={jumpKey} className={jumpKey > 0 ? "roboss-dog-jump" : ""}>
          <RobotDog running={running || activeIndex >= 0} className="h-[66px] w-[110px]" />
        </div>
      </div>
      <div className="absolute inset-x-0 bottom-0 h-10 bg-[repeating-linear-gradient(90deg,rgba(169,150,201,0.12)_0_8px,rgba(169,150,201,0.04)_8px_16px)]" />
      <div className="absolute bottom-8 left-[10%] h-5 w-5 border border-neon-green/45 bg-neon-green/20 shadow-[20px_0_0_rgba(60,242,138,0.18),10px_-10px_0_rgba(60,242,138,0.14)]" />
      <div className="absolute bottom-8 left-[46%] h-5 w-5 border border-neon-cyan/45 bg-neon-cyan/20 shadow-[20px_0_0_rgba(47,232,234,0.18),10px_-10px_0_rgba(47,232,234,0.14)]" />
      <div className="absolute bottom-8 right-[11%] h-5 w-5 border border-dashed border-sage-500/70 shadow-[20px_0_0_rgba(111,95,143,0.08),10px_-10px_0_rgba(111,95,143,0.08)]" />
    </div>
  );
}

export default function PipelineCircuitPanel({ batch }) {
  const [entries, setEntries] = useState([]);
  const [live, setLive] = useState(false);
  const [jumpKey, setJumpKey] = useState(0);
  const previousActiveIndexRef = useRef(null);

  const appendEntry = useCallback((entry) => {
    if (!entry?.id) {
      return;
    }
    setEntries((current) => {
      if (current.some((item) => item.id === entry.id)) {
        return current;
      }
      return [...current, entry].slice(-MAX_ENTRIES);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const payload = await getLogs();
        if (!cancelled) {
          const initial = Array.isArray(payload.entries) ? payload.entries : [];
          setEntries(initial.slice(-MAX_ENTRIES));
        }
      } catch {
        if (!cancelled) {
          setEntries([]);
        }
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const close = openLogStream({
      onOpen: () => setLive(true),
      onError: () => setLive(false),
      onEntry: appendEntry,
    });
    return () => {
      setLive(false);
      close();
    };
  }, [appendEntry]);

  const batchEntries = useMemo(() => {
    if (!batch?.id) {
      return [];
    }
    return entries.filter((entry) => entry.batch_id === batch.id || String(entry.message || "").includes(batch.id));
  }, [batch?.id, entries]);

  const circuit = useMemo(() => deriveCircuitState(batch, batchEntries), [batch, batchEntries]);
  const lastMessage = circuit.lastEntry?.message || "Start a generation to send the robot dog through the pipeline.";
  const batchLabel = batch?.id ? `Batch ${batch.id}` : "No active batch";
  const statusLabel = batch?.status || "idle";

  useEffect(() => {
    if (previousActiveIndexRef.current == null) {
      previousActiveIndexRef.current = circuit.activeIndex;
      return;
    }
    if (previousActiveIndexRef.current !== circuit.activeIndex) {
      previousActiveIndexRef.current = circuit.activeIndex;
      setJumpKey((value) => value + 1);
    }
  }, [circuit.activeIndex]);

  return (
    <section className="mt-6 overflow-hidden rounded-lg border border-surface-700 bg-surface-900">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-surface-700 px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-white">Pipeline circuit</h2>
          <p className="truncate text-xs text-sage-500">{batchLabel}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 font-medium ${
              live
                ? "border-neon-green/40 bg-neon-green/10 text-neon-green"
                : "border-neon-red/40 bg-neon-red/10 text-neon-red"
            }`}
          >
            {live ? <Radio size={13} aria-hidden="true" /> : <WifiOff size={13} aria-hidden="true" />}
            {live ? "Live" : "Offline"}
          </span>
          <span className="rounded-md border border-surface-600 bg-surface-950 px-2.5 py-1.5 font-medium text-sage-300">
            {statusLabel}
          </span>
        </div>
      </div>

      <div className="p-4">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-md border border-surface-700 bg-surface-950 px-3 py-2 text-xs">
          <span className="font-medium text-sage-200">
            Current step: {circuit.lastStage?.label || "Queue"}
          </span>
          <span className="min-w-0 flex-1 truncate text-right text-sage-500">{lastMessage}</span>
        </div>

        <DesktopCircuit
          activeIndex={circuit.activeIndex}
          running={circuit.running}
          statuses={circuit.statuses}
          jumpKey={jumpKey}
        />

        <div className="grid grid-cols-2 gap-3 pt-8 sm:grid-cols-4 lg:hidden">
          {STAGES.map((stage, index) => (
            <StageCard
              key={stage.key}
              stage={stage}
              index={index}
              status={circuit.statuses[index]}
              active={index === circuit.activeIndex}
              running={circuit.running}
              jumpKey={jumpKey}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
