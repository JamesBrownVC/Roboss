import { useEffect, useMemo, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth } from "../lib/api.js";

const NAV_ITEMS = [
  { to: "/", label: "Home" },
  { to: "/studio", label: "Studio" },
  { to: "/analytics", label: "Analytics" },
  { to: "/monitor", label: "Monitor" },
];

export default function AppLayout() {
  const [health, setHealth] = useState(null);
  const [checked, setChecked] = useState(false);
  const [wave, setWave] = useState({ x: 0, amp: 0 });
  const navRef = useRef(null);

  // The Home title dispatches its hovered-letter position; the divider line
  // bulges there and the nav buttons above get pushed up.
  useEffect(() => {
    function onTitleWave(event) {
      setWave(event.detail || { x: 0, amp: 0 });
    }
    window.addEventListener("roboss:titlewave", onTitleWave);
    return () => window.removeEventListener("roboss:titlewave", onTitleWave);
  }, []);

  useEffect(() => {
    const nav = navRef.current;
    if (!nav) {
      return;
    }
    const canAnimateWave = window.innerWidth >= 1024 && wave.amp > 0;
    nav.querySelectorAll("a").forEach((link) => {
      if (!canAnimateWave) {
        link.style.translate = "0 0";
        link.style.transition = "translate 160ms ease-out";
        return;
      }
      const rect = link.getBoundingClientRect();
      const center = rect.left + rect.width / 2;
      const dist = center - wave.x;
      const lift = wave.amp * 12 * Math.exp(-(dist * dist) / (2 * 300 * 300));
      // `translate` composes with Tailwind's transform-based hover lift.
      link.style.translate = `0 ${(-lift).toFixed(1)}px`;
      link.style.transition = "translate 160ms ease-out";
    });
  }, [wave]);

  const dividerPath = useMemo(() => {
    const W = 1440;
    const N = 48;
    const baseY = 13;
    const vw = typeof window !== "undefined" && window.innerWidth ? window.innerWidth : W;
    const bumpX = (wave.x / vw) * W;
    const sigma = 135;
    const points = [];
    for (let i = 0; i <= N; i += 1) {
      const px = (W / N) * i;
      const bump = wave.amp * 14 * Math.exp(-((px - bumpX) ** 2) / (2 * sigma * sigma));
      const ripple = wave.amp * 1.8 * Math.sin(px / 46);
      points.push(`${px.toFixed(1)} ${(baseY - bump + ripple).toFixed(2)}`);
    }
    return `M${points.join(" L")}`;
  }, [wave]);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      const result = await getHealth();
      if (!cancelled) {
        setHealth(result);
        setChecked(true);
      }
    }
    check();
    const timer = window.setInterval(check, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const apiOnline = Boolean(health?.ok);
  const apiReady = apiOnline && health.geminiApiKeyConfigured !== false;
  const statusLabel = !checked
    ? "Connecting"
    : !apiOnline
      ? "Offline"
      : apiReady
        ? "Connected"
        : "Missing API key";

  return (
    <div className="min-h-screen bg-transparent text-sage-100">
      <header className="sticky top-0 z-20 overflow-x-clip border-b border-surface-700 bg-black/60 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-start justify-between gap-3 px-4 pt-3 sm:items-center sm:px-6 sm:pt-4 lg:px-10">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-md border border-neon-cyan/30 bg-surface-950 shadow-[0_0_18px_rgba(241,61,245,0.42)]">
              <img
                src="/robot-hero.png"
                alt="Roboss"
                className="h-8 w-8 object-contain"
                draggable="false"
              />
            </div>
            <div className="flex min-w-0 items-baseline gap-2.5">
              <span className="shrink-0 text-[15px] font-semibold uppercase tracking-wide text-white">Roboss</span>
              <span className="hidden text-sage-500 sm:inline" aria-hidden="true">
                //
              </span>
              <span className="hidden truncate text-sm text-sage-300 sm:inline">Video Data Studio</span>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2 rounded-full border border-surface-600 px-2.5 py-1.5 text-xs font-medium text-sage-300 sm:px-3">
            <span
              className={`h-2 w-2 rounded-full ${
                !checked
                  ? "bg-sage-500"
                  : !apiOnline
                    ? "bg-[#ff3b6b] shadow-[0_0_8px_#ff3b6b]"
                    : apiReady
                      ? "bg-[#3cf28a] shadow-[0_0_8px_#3cf28a]"
                      : "bg-accent-500"
              }`}
            />
            <span>{statusLabel}</span>
          </div>
        </div>

        <nav
          ref={navRef}
          className="mx-auto grid w-full max-w-7xl grid-cols-2 gap-2 px-4 pb-2 pt-3 sm:grid-cols-4 sm:px-6 lg:flex lg:items-end lg:justify-center lg:gap-8 lg:px-10 lg:pb-0 lg:pt-4"
        >
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `min-w-0 px-3 py-3 text-center text-xs font-semibold uppercase tracking-[0.12em] transition duration-200 hover:bg-surface-900 hover:text-white sm:px-4 sm:text-sm sm:tracking-[0.16em] lg:min-w-[230px] lg:-translate-y-[3px] lg:px-12 lg:py-4 lg:text-base lg:tracking-[0.24em] lg:hover:-translate-y-[8px] ${
                  isActive
                    ? "bg-surface-850 text-white shadow-[0_0_22px_rgba(241,61,245,0.32),inset_0_1px_0_rgba(255,255,255,0.12),inset_0_-3px_0_#2fe8ea]"
                    : "text-sage-300 hover:bg-surface-900 hover:text-white hover:shadow-[0_0_18px_rgba(47,232,234,0.16),inset_0_-3px_0_rgba(47,232,234,0.35)]"
                }`
              }
              style={({ isActive }) => ({
                borderStyle: "solid",
                borderWidth: "1px 1px 0 1px",
                borderImage: isActive
                  ? "linear-gradient(90deg, #f13df5, #8b5cf6, #2fe8ea) 1"
                  : "linear-gradient(90deg, rgba(241,61,245,0.22), rgba(47,232,234,0.22)) 1",
              })}
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <svg
          aria-hidden="true"
          className="header-divider-wave pointer-events-none absolute inset-x-0 bottom-[-10px] h-6 w-full opacity-0 transition-opacity duration-200"
          viewBox="0 0 1440 24"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="header-divider-gradient" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="rgba(241,61,245,0)" />
              <stop offset="18%" stopColor="rgba(241,61,245,0.55)" />
              <stop offset="50%" stopColor="rgba(139,92,246,0.32)" />
              <stop offset="82%" stopColor="rgba(47,232,234,0.55)" />
              <stop offset="100%" stopColor="rgba(47,232,234,0)" />
            </linearGradient>
          </defs>
          <path
            d={dividerPath}
            fill="none"
            stroke="url(#header-divider-gradient)"
            strokeWidth="2"
            style={{ d: `path("${dividerPath}")`, transition: "d 160ms ease-out" }}
          />
          <path
            d={dividerPath}
            fill="none"
            stroke="rgba(47,232,234,0.18)"
            strokeWidth="7"
            style={{ d: `path("${dividerPath}")`, transition: "d 160ms ease-out" }}
          />
        </svg>
      </header>

      <main>
        <div className="mx-auto w-full max-w-7xl px-3 py-5 sm:px-6 sm:py-8 lg:px-10">
          <Outlet context={{ health }} />
        </div>
      </main>
    </div>
  );
}
