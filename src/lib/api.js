const RUNS_STORAGE_KEY = "roboss.runs.v1";

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

export function getBatch(batchId) {
  return fetchJson(`/api/batches/${batchId}`);
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
