import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, Radio, ScrollText, Trash2, WifiOff } from "lucide-react";
import { getLogs, openLogStream } from "../lib/api.js";

const MAX_VISIBLE = 500;
const LEVELS = ["all", "info", "warn", "error"];

const LEVEL_TONES = {
  info: "text-sage-200",
  warn: "text-accent-300",
  error: "text-[#ff3b6b]",
};

const AGENT_TONES = {
  intent: "border-violet-400/30 text-violet-300",
  contract: "border-indigo-400/30 text-indigo-300",
  scenarios: "border-blue-400/30 text-blue-300",
  validator: "border-accent-500/30 text-accent-300",
  compiler: "border-cyan-400/30 text-cyan-300",
  canvas: "border-pink-400/30 text-pink-300",
  omni: "border-sage-500 text-sage-200",
  veo: "border-sage-500 text-sage-200",
  verifier: "border-[#3cf28a]/40 text-[#3cf28a]",
  pipeline: "border-surface-600 text-sage-300",
  api: "border-surface-600 text-sage-200",
  system: "border-surface-600 text-sage-400",
};

function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function agentTone(agent) {
  return AGENT_TONES[agent] || AGENT_TONES.system;
}

export default function AgentLogsPanel() {
  const [entries, setEntries] = useState([]);
  const [levelFilter, setLevelFilter] = useState("all");
  const [agentFilter, setAgentFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);

  const listRef = useRef(null);
  const lastIdRef = useRef(0);

  const appendEntry = useCallback((entry) => {
    if (!entry?.id) {
      return;
    }
    lastIdRef.current = Math.max(lastIdRef.current, entry.id);
    setEntries((current) => {
      if (current.some((item) => item.id === entry.id)) {
        return current;
      }
      return [...current, entry].slice(-MAX_VISIBLE);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const payload = await getLogs();
        if (cancelled) {
          return;
        }
        const initial = Array.isArray(payload.entries) ? payload.entries : [];
        setEntries(initial.slice(-MAX_VISIBLE));
        lastIdRef.current = initial.length ? initial[initial.length - 1].id : 0;
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
  }, [reconnectKey]);

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
  }, [appendEntry, reconnectKey]);

  useEffect(() => {
    if (paused || !listRef.current) {
      return;
    }
    listRef.current.scrollTo({ top: 0, behavior: "smooth" });
  }, [entries, paused]);

  const agents = useMemo(() => {
    const unique = new Set(entries.map((entry) => entry.agent).filter(Boolean));
    return ["all", ...Array.from(unique).sort()];
  }, [entries]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return entries.filter((entry) => {
      if (levelFilter !== "all" && entry.level !== levelFilter) {
        return false;
      }
      if (agentFilter !== "all" && entry.agent !== agentFilter) {
        return false;
      }
      if (!query) {
        return true;
      }
      const haystack = [
        entry.message,
        entry.agent,
        entry.batch_id,
        entry.job_id,
        entry.level,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    }).sort((a, b) => (b.id || 0) - (a.id || 0));
  }, [entries, levelFilter, agentFilter, search]);

  function handleScroll() {
    const node = listRef.current;
    if (!node) {
      return;
    }
    const atTop = node.scrollTop < 48;
    setPaused(!atTop);
  }

  function clearLogs() {
    setEntries([]);
    lastIdRef.current = 0;
  }

  function reconnect() {
    setReconnectKey((value) => value + 1);
  }

  return (
    <section className="mt-6">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 rounded-md border border-surface-600 bg-surface-900 px-3 py-2 text-xs font-medium">
          {live ? (
            <>
              <Radio className="text-[#3cf28a]" size={14} aria-hidden="true" />
              <span className="text-[#3cf28a]">Live</span>
            </>
          ) : (
            <>
              <WifiOff className="text-[#ff3b6b]" size={14} aria-hidden="true" />
              <span className="text-[#ff3b6b]">Offline</span>
            </>
          )}
        </div>

        <select
          value={levelFilter}
          onChange={(event) => setLevelFilter(event.target.value)}
          className="h-9 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-100 outline-none transition focus:border-sage-400"
        >
          {LEVELS.map((level) => (
            <option key={level} value={level}>
              Level: {level}
            </option>
          ))}
        </select>

        <select
          value={agentFilter}
          onChange={(event) => setAgentFilter(event.target.value)}
          className="h-9 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-100 outline-none transition focus:border-sage-400"
        >
          {agents.map((agent) => (
            <option key={agent} value={agent}>
              Agent: {agent}
            </option>
          ))}
        </select>

        <input
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search messages..."
          className="h-9 min-w-[200px] flex-1 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs text-sage-100 outline-none transition placeholder:text-sage-500 focus:border-sage-400"
        />

        <button
          type="button"
          onClick={() => setPaused((value) => !value)}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-200 transition hover:border-sage-500 hover:text-white"
        >
          {paused ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}
          {paused ? "Resume scroll" : "Auto-scroll"}
        </button>

        <button
          type="button"
          onClick={clearLogs}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-200 transition hover:border-sage-500 hover:text-white"
        >
          <Trash2 size={14} aria-hidden="true" />
          Clear
        </button>

        {!live ? (
          <button
            type="button"
            onClick={reconnect}
            className="inline-flex h-9 items-center gap-2 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-3 text-xs font-medium text-[#0b0714] shadow-[0_0_16px_rgba(241,61,245,0.35)] transition hover:brightness-110"
          >
            Reconnect
          </button>
        ) : null}
      </div>

      <div className="flex min-h-[420px] flex-col rounded-lg border border-surface-700 bg-surface-900">
        <div className="flex items-center justify-between border-b border-surface-700 px-4 py-3">
          <div className="flex items-center gap-2">
            <ScrollText size={16} className="text-sage-300" aria-hidden="true" />
            <span className="text-sm font-medium text-white">Agent activity</span>
          </div>
          <span className="text-xs text-sage-500">
            {filtered.length} / {entries.length} entries
          </span>
        </div>

        <div
          ref={listRef}
          onScroll={handleScroll}
          className="flex-1 space-y-1 overflow-y-auto p-3 font-mono text-xs"
        >
          {filtered.length ? (
            filtered.map((entry) => (
              <div
                key={entry.id}
                className="grid grid-cols-[auto_auto_1fr] items-start gap-x-3 gap-y-1 rounded-md px-2 py-1.5 hover:bg-surface-850"
              >
                <span className="whitespace-nowrap text-sage-500">{formatTime(entry.ts)}</span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${agentTone(entry.agent)}`}
                >
                  {entry.agent || "system"}
                </span>
                <div className="min-w-0">
                  <p className={`leading-relaxed ${LEVEL_TONES[entry.level] || LEVEL_TONES.info}`}>
                    {entry.message}
                  </p>
                  {entry.batch_id || entry.job_id ? (
                    <p className="mt-0.5 text-[10px] text-sage-500">
                      {[entry.batch_id, entry.job_id].filter(Boolean).join(" - ")}
                    </p>
                  ) : null}
                </div>
              </div>
            ))
          ) : (
            <div className="flex h-full min-h-[320px] flex-col items-center justify-center gap-2 text-center text-sm text-sage-400">
              <ScrollText size={28} className="text-sage-500" aria-hidden="true" />
              <p>Agent logs will appear here when the backend runs a batch.</p>
              <p className="text-xs">Start a generation from Studio to see live activity.</p>
            </div>
          )}

        </div>
      </div>
    </section>
  );
}

