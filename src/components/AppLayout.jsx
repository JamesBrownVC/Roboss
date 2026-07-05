import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Bot } from "lucide-react";
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
      <header className="sticky top-0 z-20 border-b border-surface-700 bg-black/60 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 pt-4 lg:px-10">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-br from-neon-magenta to-neon-violet text-[#0b0714] shadow-[0_0_18px_rgba(241,61,245,0.6)]">
              <Bot size={18} aria-hidden="true" />
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-[15px] font-semibold uppercase tracking-wide text-white">Roboss</span>
              <span className="text-sage-500" aria-hidden="true">
                //
              </span>
              <span className="text-sm text-sage-300">Video Data Studio</span>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-surface-600 px-3 py-1.5 text-xs font-medium text-sage-300">
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

        <nav className="mx-auto flex w-full max-w-7xl items-end justify-center gap-5 px-6 pb-0 pt-4 lg:gap-8 lg:px-10">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `min-w-[230px] translate-y-px px-12 py-4 text-center text-base font-semibold uppercase tracking-[0.24em] transition duration-200 ${
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
      </header>

      <main>
        <div className="mx-auto w-full max-w-7xl px-6 py-8 lg:px-10">
          <Outlet context={{ health }} />
        </div>
      </main>
    </div>
  );
}
