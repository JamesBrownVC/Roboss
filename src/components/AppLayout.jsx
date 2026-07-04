import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Bot } from "lucide-react";
import { getHealth } from "../lib/api.js";

const NAV_ITEMS = [
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
    <div className="min-h-screen bg-surface-950 text-sage-100">
      <header className="sticky top-0 z-20 border-b border-surface-700 bg-black/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 pt-4 lg:px-10">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-white text-black">
              <Bot size={18} aria-hidden="true" />
            </div>
            <div className="flex items-baseline gap-2.5">
              <span className="text-[15px] font-semibold tracking-tight text-white">Roboss</span>
              <span className="text-sage-500" aria-hidden="true">
                /
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
                    ? "bg-[#e5484d]"
                    : apiReady
                      ? "bg-[#45a557]"
                      : "bg-accent-500"
              }`}
            />
            <span>{statusLabel}</span>
          </div>
        </div>

        <nav className="mx-auto flex w-full max-w-7xl items-center px-3 pt-2 lg:px-7">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `-mb-px border-b-2 px-3 pb-2.5 pt-2 text-sm transition ${
                  isActive
                    ? "border-white font-medium text-white"
                    : "border-transparent text-sage-300 hover:text-white"
                }`
              }
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
