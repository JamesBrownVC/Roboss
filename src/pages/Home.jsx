import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Database,
  Gauge,
  Play,
  RefreshCw,
  ScanEye,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { getStats } from "../lib/api.js";

/*
 * Ring sectors (0° = top, clockwise):
 *   A = top-left (270°→360°), B = bottom-left (180°→270°),
 *   C = bottom-right (90°→180°), D = top-right (0°→90°).
 * Each card lights the half nearest to it:
 *   Prompt-to-dataset (left)  → A+B (180°→360°)
 *   Physics verified  (right) → C+D (0°→180°)
 *   Auto-labeled     (bottom) → B+C (90°→270°)
 */
const FEATURE_CHIPS = [
  {
    icon: Sparkles,
    label: "Prompt-to-dataset",
    detail: "One prompt, N camera variants",
    color: "#f13df5",
    orbit: "lg:absolute lg:left-[-150px] lg:top-[40%]",
    arcFrom: 220,
    arcSize: 140,
  },
  {
    icon: ShieldCheck,
    label: "Physics verified",
    detail: "10 checks, 2 gates",
    color: "#2fe8ea",
    orbit: "lg:absolute lg:right-[-150px] lg:top-[40%]",
    arcFrom: 0,
    arcSize: 140,
  },
  {
    icon: ScanEye,
    label: "Auto-labeled",
    detail: "Boxes, poses, hazards",
    color: "#3cf28a",
    orbit: "lg:absolute lg:bottom-[-10px] lg:left-1/2 lg:-translate-x-1/2",
    arcFrom: 90,
    arcSize: 180,
  },
];

const TITLE_LETTERS = [
  { ch: "R", color: null },
  { ch: "O", color: null },
  { ch: "B", color: null },
  { ch: "O", color: "#f13df5" },
  { ch: "S", color: "#8b5cf6" },
  { ch: "S", color: "#2fe8ea" },
];

const RING_GRADIENT =
  "conic-gradient(from 120deg, #f13df5, #8b5cf6, #2fe8ea, #3cf28a, #f13df5)";
const RING_MASK =
  "radial-gradient(farthest-side, transparent calc(100% - 3px), black calc(100% - 2px))";
const ARC_MASK =
  "radial-gradient(farthest-side, transparent calc(100% - 8px), black calc(100% - 7px))";
const RING_TOP_FADE =
  "linear-gradient(to bottom, transparent 0%, transparent 18%, rgba(0,0,0,0.35) 30%, black 44%, black 100%)";

const PIPELINE_STAGES = [
  { key: "generated", label: "Synthesize", detail: "video jobs", x: 70, y: 50, h: 160, tone: "#2fe8ea" },
  { key: "reviewed", label: "Physics", detail: "reviewed", x: 270, y: 64, h: 132, tone: "#2fe8ea" },
  { key: "validated", label: "Safety", detail: "passed", x: 470, y: 70, h: 120, tone: "#8b5cf6" },
  { key: "labeled", label: "Label QA", detail: "annotated", x: 670, y: 74, h: 112, tone: "#f13df5" },
  { key: "committed", label: "Commit", detail: "accepted set", x: 860, y: 74, h: 112, tone: "#3cf28a" },
];

function formatNumber(value) {
  return Math.round(value).toLocaleString("en-US");
}

function formatRate(value) {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return value >= 10 ? Math.round(value).toLocaleString("en-US") : value.toFixed(1);
}

function latestBatchLabel(runs) {
  const recent = [...runs].sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")))[0];
  const id = String(recent?.id || "");
  const match = id.match(/^(.*)-job-\d+$/);
  return match?.[1] ? match[1] : id ? id.slice(0, 12) : "none";
}

function buildPipelineFromRuns(runs, source) {
  const generated = runs.length;
  const reviewed = runs.filter((run) => run.reviewStatus === "passed" || run.reviewStatus === "failed").length;
  const validated = runs.filter((run) => run.reviewStatus === "passed").length;
  const labeled = runs.filter((run) => run.labelStatus === "completed" || Number(run.zoneCount) > 0).length;
  const committed = runs.filter((run) => run.status === "completed").length;
  const validationFailed = runs.filter((run) => run.reviewStatus === "failed").length;
  const generationFailed = runs.filter((run) => run.status === "failed" && run.reviewStatus !== "failed").length;
  const labelingFailed = runs.filter((run) => run.labelStatus === "failed").length;
  const totalSeconds = runs.reduce((sum, run) => sum + (Number(run.totalSeconds) || 0), 0);
  const acceptance = generated ? Math.round((committed / generated) * 100) : null;
  const throughput = totalSeconds ? (generated / totalSeconds) * 60 : 0;

  return {
    acceptance: acceptance == null ? "pending" : `${acceptance}%`,
    batch: latestBatchLabel(runs),
    bars: [generated, reviewed, validated, labeled, committed].map(formatNumber),
    committed: formatNumber(committed),
    candidates: formatNumber(generated),
    generated,
    progressPct: generated ? Math.round((committed / generated) * 100) : 0,
    rejects: [validationFailed, generationFailed, labelingFailed].map(formatNumber),
    source,
    throughput: formatRate(throughput),
    totalRejected: formatNumber(generated - committed),
  };
}

function FeatureCard({ chip, className = "", onEnter, onLeave }) {
  const { icon: Icon, label, detail, color } = chip;
  return (
    <div
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      className={`flex items-center gap-3 px-4 py-3 backdrop-blur transition-transform duration-200 hover:-translate-y-1 ${className}`}
      style={{
        clipPath:
          "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px))",
        border: `1px solid ${color}55`,
        background: `linear-gradient(160deg, ${color}24 0%, rgba(15,10,26,0.94) 55%)`,
        boxShadow: `0 0 22px ${color}26, inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -6px 14px rgba(0,0,0,0.45)`,
      }}
    >
      <div
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md"
        style={{
          color,
          background: `${color}1f`,
          border: `1px solid ${color}66`,
          boxShadow: `0 0 12px ${color}44, inset 0 1px 0 rgba(255,255,255,0.18)`,
        }}
      >
        <Icon size={19} aria-hidden="true" />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-wide text-white">{label}</p>
        <p className="truncate text-[11px] text-sage-300">{detail}</p>
      </div>
    </div>
  );
}

function PipelineKpi({ icon: Icon, label, value, tone = "text-white", suffix }) {
  return (
    <div className="rounded-lg border border-surface-700 bg-surface-900 p-3 sm:p-4">
      <div className="flex items-center gap-2 text-sage-400">
        <Icon size={15} aria-hidden="true" />
        <span className="text-xs font-medium uppercase tracking-label">{label}</span>
      </div>
      <div className={`mt-2 text-xl font-semibold tracking-tight sm:text-2xl ${tone}`}>
        {value}
        {suffix ? <span className="ml-1 text-sm font-medium text-sage-500">{suffix}</span> : null}
      </div>
    </div>
  );
}

function MobilePipelineFlow({ pipeline }) {
  return (
    <div className="lg:hidden">
      <div className="space-y-3">
        {PIPELINE_STAGES.map((stage, index) => (
          <div key={stage.key} className="relative pl-7">
            {index < PIPELINE_STAGES.length - 1 ? (
              <span
                aria-hidden="true"
                className="absolute left-[11px] top-8 h-[calc(100%+0.75rem)] w-px bg-gradient-to-b from-neon-cyan/70 to-neon-magenta/40"
              />
            ) : null}
            <span
              aria-hidden="true"
              className="absolute left-0 top-5 h-[22px] w-[22px] rounded-md border border-neon-cyan/50 bg-surface-950 shadow-[0_0_14px_rgba(47,232,234,0.25)]"
            />
            <div className="rounded-lg border border-surface-700 bg-surface-950/80 p-3">
              <div className="flex min-w-0 items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white">{stage.label}</p>
                  <p className="text-xs text-sage-500">{stage.detail}</p>
                </div>
                <span className="font-mono text-sm font-semibold text-neon-cyan">{pipeline.bars[index]}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        {[
          { label: "validation failed", value: pipeline.rejects[0] },
          { label: "generation failed", value: pipeline.rejects[1] },
          { label: "labeling failed", value: pipeline.rejects[2] },
        ].map((reject) => (
          <div key={reject.label} className="rounded-md border border-neon-red/30 bg-neon-red/10 px-3 py-2">
            <p className="font-mono text-sm font-semibold text-neon-red">-{reject.value}</p>
            <p className="text-[11px] uppercase tracking-label text-sage-400">{reject.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [activeNode, setActiveNode] = useState(null);
  const [letterHover, setLetterHover] = useState(null);
  const [pipelineRuns, setPipelineRuns] = useState([]);
  const [pipelineSource, setPipelineSource] = useState("api");
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineUpdatedAt, setPipelineUpdatedAt] = useState(null);

  async function loadPipelineStats() {
    setPipelineLoading(true);
    try {
      const result = await getStats();
      setPipelineRuns(Array.isArray(result.runs) ? result.runs : []);
      setPipelineSource(result.source);
      setPipelineUpdatedAt(new Date());
    } finally {
      setPipelineLoading(false);
    }
  }

  useEffect(() => {
    loadPipelineStats();
    const timer = window.setInterval(loadPipelineStats, 30000);
    return () => window.clearInterval(timer);
  }, []);

  const pipeline = useMemo(
    () => buildPipelineFromRuns(pipelineRuns, pipelineSource),
    [pipelineRuns, pipelineSource],
  );
  useEffect(() => {
    document.body.classList.toggle("title-lift-active", letterHover != null);
    return () => document.body.classList.remove("title-lift-active");
  }, [letterHover]);

  function emitTitleWave(x, amp) {
    window.dispatchEvent(
      new CustomEvent("roboss:titlewave", { detail: { x, amp } }),
    );
  }

  function letterCenterX(event) {
    const rect = event?.currentTarget?.getBoundingClientRect?.();
    return rect ? rect.left + rect.width / 2 : window.innerWidth / 2;
  }

  // Wave: hovered letter rises the most, neighbours follow — the whole
  // baseline deforms as the mouse sweeps across the title.
  function letterLift(index) {
    if (letterHover == null) {
      return 0;
    }
    return Math.max(0, 30 - Math.abs(letterHover - index) * 11);
  }

  return (
    <>
    <section className="relative flex min-h-[calc(100svh-190px)] flex-col items-center justify-start overflow-hidden pb-8 pt-4 sm:min-h-[calc(100vh-140px)] sm:pb-10 sm:pt-4">
      {/* radial glow behind the robot */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-1/2 h-[420px] w-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-70 sm:h-[620px] sm:w-[620px] lg:h-[720px] lg:w-[720px]"
        style={{
          background:
            "radial-gradient(circle, rgba(241,61,245,0.28) 0%, rgba(139,92,246,0.16) 40%, transparent 68%)",
        }}
      />

      {/* giant title, sitting behind the robot — letters lift on hover */}
      <div className="relative z-0 -mb-10 select-none text-center sm:-mb-24 lg:-mb-36">
        <h1 className="font-display text-[18vw] font-black uppercase leading-none tracking-[0.03em] text-white sm:text-[13vw] sm:tracking-[0.06em] lg:text-[10rem]">
          {TITLE_LETTERS.map(({ ch, color }, index) => (
            <span
              key={index}
              onMouseEnter={(event) => {
                setLetterHover(index);
                emitTitleWave(letterCenterX(event), 1);
              }}
              onMouseLeave={(event) => {
                setLetterHover(null);
                emitTitleWave(letterCenterX(event), 0);
              }}
              className="inline-block cursor-default transition-transform duration-200 ease-out"
              style={{
                color: color || undefined,
                textShadow:
                  letterHover === index && color
                    ? `0 0 34px ${color}`
                    : letterHover === index
                      ? "0 0 34px rgba(255,255,255,0.55)"
                      : undefined,
                transform: `translateY(-${letterLift(index)}px)`,
              }}
            >
              {ch}
            </span>
          ))}
        </h1>
      </div>

      {/* robot + orbit ring + feature nodes */}
      <div className="pointer-events-none relative z-10">
        {/* base ring — whole circle, lights up when any node is hovered */}
        <div
          aria-hidden="true"
          className="absolute left-1/2 top-[42%] z-0 hidden h-[560px] w-[560px] -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-300 lg:block"
          style={{
            background: RING_GRADIENT,
            WebkitMaskImage: `${RING_MASK}, ${RING_TOP_FADE}`,
            maskImage: `${RING_MASK}, ${RING_TOP_FADE}`,
            WebkitMaskComposite: "source-in",
            maskComposite: "intersect",
            opacity: activeNode != null ? 0.85 : 0.28,
            filter:
              activeNode != null
                ? "drop-shadow(0 0 12px rgba(241,61,245,0.6))"
                : "drop-shadow(0 0 6px rgba(241,61,245,0.35))",
          }}
        />

        {/* sector arcs — wider, brighter half-ring next to the hovered card */}
        {FEATURE_CHIPS.map((chip, index) => (
          <div
            key={chip.label}
            aria-hidden="true"
            className="absolute left-1/2 top-[42%] z-0 hidden h-[584px] w-[584px] -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-300 lg:block"
            style={{
              background: `conic-gradient(from ${chip.arcFrom}deg, ${chip.color}, #8b5cf6 ${chip.arcSize / 2}deg, ${chip.color} ${chip.arcSize}deg, transparent ${chip.arcSize}deg, transparent 360deg)`,
              WebkitMaskImage: `${ARC_MASK}, ${RING_TOP_FADE}`,
              maskImage: `${ARC_MASK}, ${RING_TOP_FADE}`,
              WebkitMaskComposite: "source-in",
              maskComposite: "intersect",
              opacity: activeNode === index ? 1 : 0,
              filter: `drop-shadow(0 0 14px ${chip.color}) drop-shadow(0 0 30px ${chip.color}66)`,
            }}
          />
        ))}

        <img
          src="/robot-hero.png"
          alt="Roboss humanoid robot with neon lights"
          className="relative z-10 w-[245px] max-w-[76vw] sm:w-[420px] lg:w-[520px]"
          style={{
            filter:
              "drop-shadow(0 0 45px rgba(241,61,245,0.35)) drop-shadow(0 0 90px rgba(139,92,246,0.25))",
            maskImage: "linear-gradient(to bottom, black 72%, transparent 98%)",
            WebkitMaskImage: "linear-gradient(to bottom, black 72%, transparent 98%)",
          }}
          draggable="false"
        />

        {/* orbit nodes (desktop) */}
        {FEATURE_CHIPS.map((chip, index) => (
          <FeatureCard
            key={chip.label}
            chip={chip}
            className={`pointer-events-auto z-20 hidden w-[230px] lg:flex ${chip.orbit}`}
            onEnter={() => setActiveNode(index)}
            onLeave={() => setActiveNode(null)}
          />
        ))}
      </div>

      {/* tagline + CTA */}
      <div className="relative z-20 -mt-5 flex w-full flex-col items-center gap-3.5 text-center sm:-mt-14 lg:mt-6">
        <p className="max-w-xl rounded-lg bg-surface-950/60 px-4 py-2.5 text-sm leading-relaxed text-sage-200 backdrop-blur-sm sm:px-5 sm:text-[15px]">
          Train robots on incidents that never happened. Describe an industrial
          scenario once — Roboss generates the video, verifies the physics and
          hands back a labeled dataset.
        </p>

        <div className="flex w-full flex-col items-stretch justify-center gap-3 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:gap-4">
          <Link
            to="/studio"
            className="group inline-flex h-12 items-center justify-center gap-2.5 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-5 text-sm font-semibold uppercase tracking-wider text-[#0b0714] shadow-[0_0_28px_rgba(241,61,245,0.5)] transition hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(241,61,245,0.7)] hover:brightness-110 sm:px-7"
          >
            <Play size={16} aria-hidden="true" />
            Generate dataset
            <ArrowRight
              size={16}
              className="transition-transform group-hover:translate-x-1"
              aria-hidden="true"
            />
          </Link>
        </div>
      </div>

      {/* feature cards fallback (mobile / tablet) */}
      <div className="relative z-20 mt-9 grid w-full max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3 lg:hidden">
        {FEATURE_CHIPS.map((chip) => (
          <FeatureCard key={chip.label} chip={chip} />
        ))}
      </div>
    </section>
    <section className="relative flex flex-col justify-center overflow-hidden py-10 sm:py-12 lg:min-h-[calc(100vh-120px)]">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-[-20%] top-10 h-px bg-gradient-to-r from-transparent via-neon-cyan/60 to-transparent"
      />

      <div className="mb-7 flex flex-col items-stretch justify-between gap-5 sm:flex-row sm:flex-wrap sm:items-end">
        <div className="max-w-2xl">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-surface-600 bg-surface-950/70 px-3 py-1.5 text-xs font-medium uppercase tracking-label text-neon-cyan">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                pipelineSource === "api"
                  ? "bg-neon-green shadow-[0_0_8px_rgba(60,242,138,0.8)]"
                  : "bg-accent-400"
              }`}
            />
            {pipelineSource === "api" ? "Backend pipeline data" : "Local pipeline cache"}
          </div>
          <h2 className="font-display text-2xl font-semibold uppercase tracking-wide text-white sm:text-4xl">
            Generation to validation to commit
          </h2>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-sage-300">
            One seed scenario expands into verified training candidates, filtered through staged
            physics, safety and label checks before entering the dataset.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 sm:justify-end">
          <button
            type="button"
            onClick={loadPipelineStats}
            disabled={pipelineLoading}
            className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-surface-600 bg-surface-950/70 px-4 text-sm font-medium text-sage-200 transition hover:border-neon-cyan hover:text-white sm:w-auto"
          >
            <RefreshCw size={15} className={pipelineLoading ? "animate-spin" : ""} aria-hidden="true" />
            Refresh data
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-surface-700 bg-surface-900 p-3 shadow-[0_0_40px_rgba(47,232,234,0.08)] sm:p-5">
        <div className="mb-3 flex flex-col items-stretch justify-between gap-4 border-b border-surface-700 pb-4 sm:flex-row sm:flex-wrap sm:items-center">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-white">Dataset funnel</h3>
            <p className="mt-1 break-words font-mono text-xs text-sage-500">
              Batch {pipeline.batch} / {pipeline.progressPct}% committed
              {pipelineUpdatedAt ? ` / updated ${pipelineUpdatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : ""}
            </p>
          </div>
          <div className="h-1.5 w-full max-w-[260px] overflow-hidden rounded-full bg-surface-700">
            <div
              className="h-full rounded-full bg-gradient-to-r from-neon-magenta via-neon-violet to-neon-cyan transition-all duration-150"
              style={{ width: `${pipeline.progressPct}%` }}
            />
          </div>
        </div>

        <MobilePipelineFlow pipeline={pipeline} />

        <div className="hidden lg:block">
          <svg viewBox="0 0 1140 282" className="w-full min-w-0">
            <defs>
              <linearGradient id="homePipelineBlue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2fe8ea" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.75" />
              </linearGradient>
              <linearGradient id="homePipelineGreen" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3cf28a" stopOpacity="0.95" />
                <stop offset="100%" stopColor="#2fe8ea" stopOpacity="0.68" />
              </linearGradient>
              <marker id="homePipelineArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6" fill="none" stroke="#8a7ab3" strokeWidth="1.4" />
              </marker>
            </defs>

            <polygon points="96,50 244,64 244,196 96,210" fill="rgba(47,232,234,0.08)" />
            <polygon points="296,64 444,70 444,190 296,196" fill="rgba(139,92,246,0.10)" />
            <polygon points="496,70 644,74 644,186 496,190" fill="rgba(241,61,245,0.08)" />
            <polygon points="696,74 834,74 834,186 696,186" fill="rgba(60,242,138,0.08)" />
            <line
              x1="96"
              y1="130"
              x2="880"
              y2="130"
              stroke="#2fe8ea"
              strokeWidth="1.4"
              strokeDasharray="2 12"
              strokeLinecap="round"
              opacity="0.65"
              className="home-pipeline-flow"
            />

            {PIPELINE_STAGES.map((stage, index) => (
              <g key={stage.label}>
                <rect
                  x={stage.x - 26}
                  y={stage.y}
                  width="52"
                  height={stage.h}
                  rx="6"
                  fill={index === PIPELINE_STAGES.length - 1 ? "url(#homePipelineGreen)" : "url(#homePipelineBlue)"}
                  stroke={stage.tone}
                  strokeOpacity="0.42"
                />
                <text
                  x={stage.x}
                  y={134}
                  textAnchor="middle"
                  fill="#f4ecff"
                  fontFamily="JetBrains Mono, ui-monospace, monospace"
                  fontSize="13"
                  fontWeight="600"
                >
                  {pipeline.bars[index]}
                </text>
                <text x={stage.x} y={34 + index * 6} textAnchor="middle" fill={stage.tone} fontSize="12" fontWeight="700">
                  {stage.label}
                </text>
                <text
                  x={stage.x}
                  y={48 + index * 5}
                  textAnchor="middle"
                  fill="#8a7ab3"
                  fontFamily="JetBrains Mono, ui-monospace, monospace"
                  fontSize="9.5"
                >
                  {stage.detail}
                </text>
              </g>
            ))}

            {[
              { x: 170, label: "generation failed", value: pipeline.rejects[1] },
              { x: 370, label: "validation failed", value: pipeline.rejects[0] },
              { x: 570, label: "labeling failed", value: pipeline.rejects[2] },
            ].map((reject) => (
              <g key={reject.label}>
                <polygon points={`${reject.x - 5},214 ${reject.x + 5},214 ${reject.x},222`} fill="#ff3b6b" />
                <text
                  x={reject.x}
                  y="238"
                  textAnchor="middle"
                  fill="#ff3b6b"
                  fontFamily="JetBrains Mono, ui-monospace, monospace"
                  fontSize="12"
                  fontWeight="600"
                >
                  -{reject.value}
                </text>
                <text
                  x={reject.x}
                  y="252"
                  textAnchor="middle"
                  fill="#8a7ab3"
                  fontFamily="JetBrains Mono, ui-monospace, monospace"
                  fontSize="9"
                >
                  {reject.label}
                </text>
              </g>
            ))}

            <line x1="888" y1="130" x2="960" y2="130" stroke="#8a7ab3" strokeWidth="1.4" markerEnd="url(#homePipelineArrow)" />
            <text x="924" y="122" textAnchor="middle" fill="#8a7ab3" fontFamily="JetBrains Mono, ui-monospace, monospace" fontSize="9">
              deploy
            </text>
            <rect x="966" y="94" width="146" height="72" rx="12" fill="#0f0a1a" stroke="rgba(47,232,234,0.35)" />
            <rect x="982" y="118" width="24" height="24" rx="5" fill="none" stroke="#c9b8ea" strokeWidth="1.6" />
            <circle cx="989" cy="127" r="1.6" fill="#2fe8ea" />
            <circle cx="999" cy="127" r="1.6" fill="#2fe8ea" />
            <text x="1020" y="126" textAnchor="start" fill="#f4ecff" fontSize="12.5" fontWeight="700">
              Simulation
            </text>
            <text x="1020" y="143" textAnchor="start" fill="#8a7ab3" fontFamily="JetBrains Mono, ui-monospace, monospace" fontSize="9">
              self-diagnosis
            </text>
          </svg>
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <PipelineKpi icon={Database} label="Backend runs" value={pipeline.candidates} />
        <PipelineKpi icon={CheckCircle2} label="Acceptance rate" value={pipeline.acceptance} tone="text-neon-green" />
        <PipelineKpi icon={ShieldCheck} label="Rejected runs" value={pipeline.totalRejected} tone="text-neon-red" />
        <PipelineKpi icon={Gauge} label="Throughput" value={pipeline.throughput} suffix="jobs/min" />
        <PipelineKpi icon={Activity} label="Dataset committed" value={pipeline.committed} tone="text-neon-cyan" />
      </div>
    </section>
    </>
  );
}
