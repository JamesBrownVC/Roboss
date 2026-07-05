import { useState } from "react";
import { Link } from "react-router-dom";
import { Activity, ArrowRight, Play, ScanEye, ShieldCheck, Sparkles } from "lucide-react";

const FEATURE_CHIPS = [
  {
    icon: Sparkles,
    label: "Prompt-to-dataset",
    detail: "One prompt, N camera variants",
    color: "#f13df5",
    orbit: "lg:absolute lg:left-[-150px] lg:top-[34%]",
  },
  {
    icon: ShieldCheck,
    label: "Physics verified",
    detail: "10 checks, 2 gates",
    color: "#2fe8ea",
    orbit: "lg:absolute lg:right-[-150px] lg:top-[34%]",
  },
  {
    icon: ScanEye,
    label: "Auto-labeled",
    detail: "Boxes, poses, hazards",
    color: "#3cf28a",
    orbit: "lg:absolute lg:bottom-[-10px] lg:left-1/2 lg:-translate-x-1/2",
  },
];

const RING_GRADIENT =
  "conic-gradient(from 120deg, #f13df5, #8b5cf6, #2fe8ea, #3cf28a, #f13df5)";
const RING_MASK =
  "radial-gradient(farthest-side, transparent calc(100% - 3px), black calc(100% - 2px))";
const RING_TOP_FADE =
  "linear-gradient(to bottom, transparent 0%, transparent 18%, rgba(0,0,0,0.35) 30%, black 44%, black 100%)";

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

export default function Home() {
  const [nodeHovered, setNodeHovered] = useState(false);

  return (
    <section className="relative -mt-6 flex min-h-[calc(100vh-140px)] flex-col items-center justify-start overflow-hidden pb-10 pt-1">
      {/* radial glow behind the robot */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-1/2 h-[720px] w-[720px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-70"
        style={{
          background:
            "radial-gradient(circle, rgba(241,61,245,0.28) 0%, rgba(139,92,246,0.16) 40%, transparent 68%)",
        }}
      />

      {/* giant title, sitting behind the robot */}
      <div className="relative z-0 -mb-16 select-none text-center sm:-mb-24 lg:-mb-36">
        <h1 className="font-display text-[17vw] font-black uppercase leading-none tracking-[0.06em] text-white sm:text-[13vw] lg:text-[10rem]">
          Rob
          <span className="bg-gradient-to-r from-neon-magenta to-neon-cyan bg-clip-text text-transparent">
            oss
          </span>
        </h1>
      </div>

      {/* robot + orbit ring + feature nodes */}
      <div className="relative z-10">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-[42%] z-0 hidden h-[560px] w-[560px] -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-300 lg:block"
          style={{
            background: RING_GRADIENT,
            WebkitMaskImage: `${RING_MASK}, ${RING_TOP_FADE}`,
            maskImage: `${RING_MASK}, ${RING_TOP_FADE}`,
            WebkitMaskComposite: "source-in",
            maskComposite: "intersect",
            opacity: nodeHovered ? 0.95 : 0.28,
            filter: nodeHovered
              ? "drop-shadow(0 0 16px rgba(241,61,245,0.8)) drop-shadow(0 0 30px rgba(47,232,234,0.5))"
              : "drop-shadow(0 0 6px rgba(241,61,245,0.35))",
          }}
        />

        <img
          src="/robot-hero.png"
          alt="Roboss humanoid robot with neon lights"
          className="relative z-10 w-[320px] max-w-[80vw] sm:w-[420px] lg:w-[520px]"
          style={{
            filter:
              "drop-shadow(0 0 45px rgba(241,61,245,0.35)) drop-shadow(0 0 90px rgba(139,92,246,0.25))",
            maskImage: "linear-gradient(to bottom, black 72%, transparent 98%)",
            WebkitMaskImage: "linear-gradient(to bottom, black 72%, transparent 98%)",
          }}
          draggable="false"
        />

        {/* orbit nodes (desktop) */}
        {FEATURE_CHIPS.map((chip) => (
          <FeatureCard
            key={chip.label}
            chip={chip}
            className={`z-20 hidden w-[230px] lg:flex ${chip.orbit}`}
            onEnter={() => setNodeHovered(true)}
            onLeave={() => setNodeHovered(false)}
          />
        ))}
      </div>

      {/* tagline + CTA */}
      <div className="relative z-20 -mt-10 flex flex-col items-center gap-3.5 text-center sm:-mt-14 lg:mt-6">
        <p className="max-w-xl rounded-lg bg-surface-950/60 px-5 py-2.5 text-sm leading-relaxed text-sage-200 backdrop-blur-sm sm:text-[15px]">
          Train robots on incidents that never happened. Describe an industrial
          scenario once — Roboss generates the video, verifies the physics and
          hands back a labeled dataset.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link
            to="/studio"
            className="group inline-flex h-12 items-center gap-2.5 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-7 text-sm font-semibold uppercase tracking-wider text-[#0b0714] shadow-[0_0_28px_rgba(241,61,245,0.5)] transition hover:shadow-[0_0_40px_rgba(241,61,245,0.7)] hover:brightness-110"
          >
            <Play size={16} aria-hidden="true" />
            Generate dataset
            <ArrowRight
              size={16}
              className="transition-transform group-hover:translate-x-1"
              aria-hidden="true"
            />
          </Link>
          <Link
            to="/monitor"
            className="inline-flex h-12 items-center gap-2.5 rounded-md border border-surface-600 bg-surface-950/60 px-6 text-sm font-medium uppercase tracking-wider text-sage-200 backdrop-blur transition hover:border-neon-cyan hover:text-white hover:shadow-[0_0_18px_rgba(47,232,234,0.25)]"
          >
            <Activity size={16} aria-hidden="true" />
            Live monitor
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
  );
}
