const RUNS_STORAGE_KEY = "roboss.runs.v1";
const ACTIVE_BATCH_STORAGE_KEY = "roboss.activeBatch.v1";
const STUDIO_STATE_STORAGE_KEY = "roboss.studioState.v1";

export async function readApiError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail) && payload.detail[0]?.msg) {
      return payload.detail[0].msg;
    }
    return payload.error || "Request failed.";
  } catch {
    return "Request failed.";
  }
}

export async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  return response.json();
}

export function createBatch({ prompt, aspectRatio, count, reference }) {
  const body = {
    prompt,
    aspect_ratio: aspectRatio,
    count,
  };
  if (reference?.kind === "image") {
    body.reference_image = { data: reference.data, mimeType: reference.mimeType };
  } else if (reference?.kind === "video") {
    body.reference_video = { data: reference.data, mimeType: reference.mimeType };
  }
  return fetchJson("/api/videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function createDogDemoBatch() {
  return fetchJson("/api/demo/dog", {
    method: "POST",
  });
}

export function createCacheDemoBatch() {
  return fetchJson("/api/demo/cache", {
    method: "POST",
  });
}

export function getBatch(batchId) {
  return fetchJson(`/api/batches/${batchId}`);
}

export function getBatchDownloadUrl(batchId) {
  return `/api/batches/${encodeURIComponent(batchId)}/download`;
}

export function readActiveBatchId() {
  try {
    return window.localStorage.getItem(ACTIVE_BATCH_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveActiveBatchId(batchId) {
  try {
    window.localStorage.setItem(ACTIVE_BATCH_STORAGE_KEY, batchId);
  } catch {
    /* active batch resume is best-effort */
  }
}

export function clearActiveBatchId(batchId) {
  try {
    const current = window.localStorage.getItem(ACTIVE_BATCH_STORAGE_KEY);
    if (!batchId || current === batchId) {
      window.localStorage.removeItem(ACTIVE_BATCH_STORAGE_KEY);
    }
  } catch {
    /* active batch resume is best-effort */
  }
}

export function readStudioState() {
  try {
    const raw = window.localStorage.getItem(STUDIO_STATE_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function saveStudioState(patch) {
  try {
    const next = { ...readStudioState(), ...patch, updatedAt: Date.now() };
    for (const key of Object.keys(next)) {
      if (next[key] == null) {
        delete next[key];
      }
    }
    window.localStorage.setItem(STUDIO_STATE_STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* Studio resume is best-effort */
  }
}

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
    return AbortSignal.timeout(ms);
  }
  return undefined;
}

export async function getHealth() {
  try {
    return await fetchJson("/api/health", { signal: timeoutSignal(8000) });
  } catch {
    return null;
  }
}

/* ---------- Stats: backend endpoint first, local history as fallback ---------- */

function readLocalRuns() {
  try {
    const raw = window.localStorage.getItem(RUNS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeLocalRuns(runs) {
  try {
    window.localStorage.setItem(RUNS_STORAGE_KEY, JSON.stringify(runs.slice(-500)));
  } catch {
    /* storage full or unavailable: stats history is best-effort */
  }
}

export function recordRuns(records) {
  if (!records.length) {
    return;
  }
  const runs = readLocalRuns();
  const knownIds = new Set(runs.map((run) => run.id));
  const fresh = records.filter((record) => record.id && !knownIds.has(record.id));
  if (fresh.length) {
    writeLocalRuns([...runs, ...fresh]);
  }
}

export async function getStats() {
  try {
    const response = await fetch("/api/stats", { signal: timeoutSignal(5000) });
    if (response.ok) {
      const payload = await response.json();
      if (payload && Array.isArray(payload.runs)) {
        return { source: "api", runs: payload.runs };
      }
    }
  } catch {
    /* backend stats endpoint unavailable: fall back to local history */
  }
  return { source: "local", runs: readLocalRuns() };
}

/* ---------- Agent logs: REST catch-up + SSE live stream ---------- */

export async function getLogs(sinceId) {
  const query = sinceId != null ? `?since=${sinceId}` : "";
  return fetchJson(`/api/logs${query}`);
}

export function openLogStream({ onEntry, onError, onOpen }) {
  const source = new EventSource("/api/logs/stream");

  source.onopen = () => {
    if (onOpen) {
      onOpen();
    }
  };

  source.onmessage = (event) => {
    try {
      const entry = JSON.parse(event.data);
      onEntry(entry);
    } catch {
      /* ignore malformed SSE payloads */
    }
  };

  source.onerror = () => {
    if (onError) {
      onError();
    }
  };

  return () => {
    source.close();
  };
}
