"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types — mirror of the backend label schema (the front/back contract)
// ---------------------------------------------------------------------------

type Track = "object" | "action" | "audio";

type Box = [number, number, number, number]; // [ymin, xmin, ymax, xmax] 0-1000

interface BoxKeyframe {
  t: number;
  box_2d: Box;
}

interface Label {
  t_start: number;
  t_end: number;
  track: Track;
  label: string;
  detail: string;
  confidence: number;
  boxes?: BoxKeyframe[] | null;
}

interface Summary {
  objects: number;
  actions: number;
  audio: number;
}

interface Version {
  id: string;
  prompt: string;
  labels: Label[];
  summary: Summary | null;
}

type Phase = "compose" | "generating" | "labeling" | "review";

const TAU = 90;
const TRACKS: Track[] = ["object", "action", "audio"];
const TRACK_META: Record<Track, { name: string; icon: string }> = {
  object: { name: "Objects", icon: "◼" },
  action: { name: "Actions", icon: "▶" },
  audio: { name: "Audio", icon: "♪" },
};

const STEPS = ["Prompt", "Generate", "Label", "Explore"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function readNdjson(res: Response, onEvent: (event: any) => void): Promise<void> {
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? body?.error ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
}

const labelKey = (l: Label) => `${l.track}:${l.label}:${l.t_start}`;

// Box at time t: linear interpolation between the surrounding keyframes
function boxAt(keyframes: BoxKeyframe[], t: number): Box | null {
  if (!keyframes.length) return null;
  const sorted = [...keyframes].sort((a, b) => a.t - b.t);
  if (t <= sorted[0].t) return sorted[0].box_2d;
  if (t >= sorted[sorted.length - 1].t) return sorted[sorted.length - 1].box_2d;
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (t >= a.t && t <= b.t) {
      const r = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
      return a.box_2d.map((v, j) => v + (b.box_2d[j] - v) * r) as Box;
    }
  }
  return sorted[0].box_2d;
}

const fmt = (t: number) => `${t.toFixed(1)}s`;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [phase, setPhase] = useState<Phase>("compose");
  const [status, setStatus] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [versions, setVersions] = useState<Version[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(10);
  const [showBoxes, setShowBoxes] = useState(true);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);

  // Track playback time at display refresh rate so boxes follow smoothly
  useEffect(() => {
    let raf: number;
    const tick = () => {
      if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [currentId]);

  const current = versions.find((v) => v.id === currentId) ?? null;
  const busy = phase === "generating" || phase === "labeling";
  const progress = Math.min(99, 100 * (1 - Math.exp(-elapsed / TAU)));
  const stepIndex = { compose: 0, generating: 1, labeling: 2, review: 3 }[phase];

  const patchVersion = (id: string, patch: Partial<Version> | ((v: Version) => Partial<Version>)) =>
    setVersions((prev) =>
      prev.map((v) =>
        v.id === id ? { ...v, ...(typeof patch === "function" ? patch(v) : patch) } : v
      )
    );

  // ---------------------------------------------------------------- labeling

  const labelVideo = useCallback(async (videoId: string) => {
    setPhase("labeling");
    setStatus("Starting agentic labeling...");
    patchVersion(videoId, { labels: [], summary: null });
    const res = await fetch("/api/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
    });
    await readNdjson(res, (event) => {
      switch (event.type) {
        case "pass_start":
          setStatus(event.message);
          setElapsed(0);
          break;
        case "progress":
          setElapsed(event.elapsed);
          if (event.message) setStatus(event.message);
          break;
        case "labels":
          patchVersion(videoId, (v) => {
            // tracking labels replace their inventory counterparts (by name)
            if (event.pass === "tracking") {
              const trackedNames = new Set((event.labels as Label[]).map((l) => l.label));
              return {
                labels: [
                  ...v.labels.filter((l) => !(l.track === "object" && trackedNames.has(l.label))),
                  ...event.labels,
                ],
              };
            }
            return { labels: [...v.labels, ...event.labels] };
          });
          break;
        case "done":
          patchVersion(videoId, { labels: event.labels, summary: event.summary });
          break;
        case "error":
          throw new Error(event.error);
      }
    });
    setPhase("review");
    setStatus("");
  }, []);

  // --------------------------------------------------------------- generate

  async function generate() {
    setPhase("generating");
    setStatus("Generating video...");
    setElapsed(0);
    setError("");
    setSelectedKey(null);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, aspectRatio: "16:9" }),
      });
      let videoId: string | null = null;
      await readNdjson(res, (event) => {
        switch (event.type) {
          case "status":
            setStatus(event.message);
            break;
          case "progress":
            setElapsed(event.elapsed);
            break;
          case "done":
            videoId = event.video_id;
            break;
          case "error":
            throw new Error(event.error);
        }
      });
      if (!videoId) throw new Error("No video id returned");
      // backend restarts reset ids (v1, v2...): drop any stale version with the same id
      setVersions((prev) => [
        ...prev.filter((v) => v.id !== videoId),
        { id: videoId!, prompt, labels: [], summary: null },
      ]);
      setCurrentId(videoId);
      await labelVideo(videoId);
    } catch (err: any) {
      setPhase(versions.length ? "review" : "compose");
      setError(err?.message ?? String(err));
    }
  }

  // -------------------------------------------------------------------- mic

  async function toggleMic() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setRecording(false);
      setTranscribing(true);
      try {
        const form = new FormData();
        form.append("audio", new Blob(chunks, { type: recorder.mimeType }), "mic.webm");
        const res = await fetch("/api/transcribe", { method: "POST", body: form });
        if (!res.ok) throw new Error(`Transcription failed (HTTP ${res.status})`);
        const { text } = await res.json();
        setPrompt((p) => (p ? `${p} ${text}` : text));
      } catch (err: any) {
        setError(err?.message ?? String(err));
      } finally {
        setTranscribing(false);
      }
    };
    recorderRef.current = recorder;
    recorder.start();
    setRecording(true);
  }

  // ------------------------------------------------------------------ player

  const selectLabel = (l: Label) => {
    setSelectedKey((k) => (k === labelKey(l) ? null : labelKey(l)));
    if (videoRef.current) {
      videoRef.current.currentTime = l.t_start + 0.05;
      videoRef.current.pause();
    }
  };

  const activeBoxes = (
    current?.labels.filter(
      (l) =>
        l.track === "object" &&
        l.boxes?.length &&
        currentTime >= l.t_start &&
        currentTime <= l.t_end &&
        (showBoxes || labelKey(l) === selectedKey)
    ) ?? []
  )
    .map((l) => ({ label: l, box: boxAt(l.boxes!, currentTime) }))
    .filter((b): b is { label: Label; box: Box } => b.box !== null);

  const timelineEnd = Math.max(duration, ...(current?.labels.map((l) => l.t_end) ?? [0]));

  const seekFromLane = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * timelineEnd;
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  // -------------------------------------------------------------------- render

  return (
    <main className="container">
      <header className="topbar">
        <div>
          <h1>Roboss</h1>
          <p className="subtitle">Generate a video, get every element labeled — image & sound</p>
        </div>
        <ol className="stepper">
          {STEPS.map((s, i) => (
            <li
              key={s}
              className={i < stepIndex ? "done" : i === stepIndex ? "active" : ""}
            >
              <span className="step-dot">{i < stepIndex ? "✓" : i + 1}</span>
              {s}
            </li>
          ))}
        </ol>
      </header>

      {/* Compose bar */}
      <section className="compose">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder='Describe the video to generate — e.g. "A forklift crosses a warehouse, an operator walks nearby. Single continuous shot."'
          rows={current ? 2 : 3}
          disabled={busy}
        />
        <div className="compose-actions">
          <button
            className={`mic ${recording ? "recording" : ""}`}
            onClick={toggleMic}
            disabled={busy || transcribing}
            title="Dictate the prompt (speech-to-text)"
          >
            {recording ? "■ Stop recording" : transcribing ? "Transcribing…" : "🎤 Dictate"}
          </button>
          <button onClick={generate} disabled={busy || !prompt.trim()}>
            {busy ? "Working…" : "Generate video"}
          </button>
        </div>
      </section>

      {/* Progress */}
      {busy && (
        <section className="progress-zone">
          <div className="progress-bar">
            <div
              className={`progress-fill ${phase === "labeling" ? "labeling" : ""}`}
              style={{ width: phase === "labeling" ? "100%" : `${progress}%` }}
            />
          </div>
          <p className="status">
            {status}
            {elapsed > 0 && <span className="elapsed"> · {elapsed}s</span>}
          </p>
        </section>
      )}

      {error && <p className="error">⚠ {error}</p>}

      {/* Workspace: player + elements panel */}
      {current && (
        <section className="workspace">
          <div className="left-col">
            <div className="player-wrap">
              <video
                ref={videoRef}
                src={`/api/videos/${current.id}`}
                controls
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 10)}
              />
              <div className="boxes-layer">
                {activeBoxes.map(({ label: l, box }) => {
                  const [ymin, xmin, ymax, xmax] = box;
                  return (
                    <div
                      key={labelKey(l)}
                      className={`bbox ${selectedKey === labelKey(l) ? "selected" : ""}`}
                      style={{
                        top: `${ymin / 10}%`,
                        left: `${xmin / 10}%`,
                        width: `${(xmax - xmin) / 10}%`,
                        height: `${(ymax - ymin) / 10}%`,
                      }}
                    >
                      <span className="bbox-tag">{l.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="player-meta">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showBoxes}
                  onChange={(e) => setShowBoxes(e.target.checked)}
                />
                Show bounding boxes
              </label>
              <div className="meta-right">
                <button
                  className="relabel"
                  onClick={() => labelVideo(current.id).catch((err) => {
                    setPhase("review");
                    setError(err?.message ?? String(err));
                  })}
                  disabled={busy}
                  title="Run the labeling passes again on this video"
                >
                  ↻ Re-label
                </button>
                <span className="synthid">SynthID watermarked · {current.id}</span>
              </div>
            </div>

            {/* Timeline */}
            {current.labels.length > 0 && (
              <div className="timeline">
                <div className="ruler" onClick={seekFromLane}>
                  {Array.from({ length: Math.floor(timelineEnd) + 1 }, (_, s) => (
                    <span
                      key={s}
                      className="tick"
                      style={{ left: `${(s / timelineEnd) * 100}%` }}
                    >
                      {s}s
                    </span>
                  ))}
                </div>
                {TRACKS.map((track) => (
                  <div key={track} className="track">
                    <span className="track-name">{TRACK_META[track].name}</span>
                    <div className="track-lane" onClick={seekFromLane}>
                      {current.labels
                        .filter((l) => l.track === track)
                        .map((l) => (
                          <button
                            key={labelKey(l)}
                            className={`segment ${track} ${
                              selectedKey === labelKey(l) ? "selected" : ""
                            }`}
                            style={{
                              left: `${(l.t_start / timelineEnd) * 100}%`,
                              width: `${Math.max(2, ((l.t_end - l.t_start) / timelineEnd) * 100)}%`,
                            }}
                            title={`${l.label} (${fmt(l.t_start)}–${fmt(l.t_end)}) · ${l.detail}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              selectLabel(l);
                            }}
                          >
                            {l.label}
                          </button>
                        ))}
                    </div>
                  </div>
                ))}
                <div
                  className="playhead"
                  style={{ left: `calc(88px + (100% - 88px) * ${currentTime / timelineEnd})` }}
                />
              </div>
            )}
          </div>

          {/* Elements panel */}
          <aside className="panel">
            <div className="panel-head">
              <h2>Elements</h2>
              {current.summary && (
                <span className="panel-count">
                  {current.summary.objects + current.summary.actions + current.summary.audio}
                </span>
              )}
            </div>
            {phase === "labeling" && (
              <p className="hint">Labels appear live as each analysis pass completes…</p>
            )}
            {phase === "review" && current.labels.length === 0 && (
              <p className="hint">No elements were detected in this video.</p>
            )}

            {TRACKS.map((track) => {
              const items = current.labels
                .filter((l) => l.track === track)
                .sort((a, b) => a.t_start - b.t_start);
              if (!items.length) return null;
              return (
                <div key={track} className="group">
                  <h3 className={`group-title ${track}`}>
                    <span className="dot" />
                    {TRACK_META[track].name}
                    <span className="group-count">{items.length}</span>
                  </h3>
                  <ul className="rows">
                    {items.map((l) => {
                      const key = labelKey(l);
                      const isSel = selectedKey === key;
                      return (
                        <li key={key}>
                          <button
                            className={`row ${track} ${isSel ? "selected" : ""}`}
                            onClick={() => selectLabel(l)}
                            title="Click to jump to this element in the video"
                          >
                            <span className="row-name">{l.label.replaceAll("_", " ")}</span>
                            <span className="row-time">
                              {fmt(l.t_start)}–{fmt(l.t_end)}
                            </span>
                            <span
                              className="row-conf"
                              title={`confidence ${l.confidence.toFixed(2)}`}
                            >
                              <span
                                className="row-conf-fill"
                                style={{ width: `${l.confidence * 100}%` }}
                              />
                            </span>
                          </button>
                          {isSel && (
                            <div className="row-detail">
                              <p>{l.detail}</p>
                              <span>confidence {l.confidence.toFixed(2)}</span>
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })}
          </aside>
        </section>
      )}

      {/* Version strip */}
      {versions.length > 1 && (
        <footer className="version-strip">
          <span className="strip-label">History</span>
          {versions.map((v) => (
            <button
              key={v.id}
              className={`version ${v.id === currentId ? "active" : ""}`}
              onClick={() => {
                setCurrentId(v.id);
                setSelectedKey(null);
              }}
              disabled={busy}
              title={v.prompt}
            >
              {v.id}
              {v.summary
                ? ` · ${v.summary.objects + v.summary.actions + v.summary.audio} labels`
                : " · …"}
            </button>
          ))}
        </footer>
      )}
    </main>
  );
}
