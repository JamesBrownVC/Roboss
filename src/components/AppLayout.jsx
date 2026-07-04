import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { BarChart3, Bot, Clapperboard, Cctv } from "lucide-react";
import { getHealth } from "../lib/api.js";

const NAV_ITEMS = [
  { to: "/studio", label: "Studio", icon: Clapperboard },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/monitor", label: "Monitor", icon: Cctv },
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

  return (
    <div className="flex min-h-screen bg-surface-950 text-sage-100">
      <aside className="fixed inset-y-0 left-0 z-20 flex w-60 flex-col border-r border-white/5 bg-surface-900">
        <div className="flex items-center gap-3 px-5 py-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-500 text-surface-950 shadow-glow">
            <Bot size={22} aria-hidden="true" />
          </div>
          <div>
            <div className="font-display text-lg font-bold tracking-tight text-white">Roboss</div>
            <div className="text-xs font-medium text-sage-300/60">Video Data Studio</div>
          </div>
        </div>

        <nav className="mt-2 flex flex-col gap-1 px-3">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg border-l-2 px-3 py-2.5 text-sm font-semibold transition ${
                  isActive
                    ? "border-accent-500 bg-white/[0.04] text-white"
                    : "border-transparent text-sage-300/70 hover:bg-white/[0.03] hover:text-sage-100"
                }`
              }
            >
              <Icon size={18} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto px-5 py-5">
          <div className="flex items-center gap-2 text-xs font-medium text-sage-300/50">
            <span
              className={`h-2 w-2 rounded-full ${
                !checked ? "bg-sage-500/50" : apiOnline ? "bg-emerald-400" : "bg-red-400"
              }`}
            />
            <span>{!checked ? "Connecting" : apiOnline ? "Connected" : "Offline"}</span>
          </div>
        </div>
      </aside>

      <main className="ml-60 flex-1">
        <div className="mx-auto w-full max-w-7xl px-6 py-8 lg:px-10">
          <Outlet context={{ health }} />
        </div>
      </main>
    </div>
  );
}
