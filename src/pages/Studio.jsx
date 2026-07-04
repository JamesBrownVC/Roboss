import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Download, ImagePlus, Info, LoaderCircle, Play, X } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import AgentLogsPanel from "../components/AgentLogsPanel.jsx";
import PageHeader from "../components/PageHeader.jsx";
import VideoCard, { STATUS_LABELS, annotationCount } from "../components/VideoCard.jsx";
import {
  clearActiveBatchId,
  createBatch,
  getBatch,
  getBatchDownloadUrl,
  readActiveBatchId,
  readStudioState,
  recordRuns,
  saveActiveBatchId,
  saveStudioState,
} from "../lib/api.js";

const POLL_INTERVAL_MS = 3000;
const PREVIEW_LIMIT = 4;
const MIN_DATASET = 1;
const DEFAULT_DATASET_COUNT = 1;
const MAX_DATASET = 5000;
const TERMINAL_STATUSES = ["completed", "failed", "partial"];
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_VIDEO_BYTES = 50 * 1024 * 1024;

const DEFAULT_PROMPT =
  "A loaded wooden pallet sits slightly unstable on a lower warehouse rack, boxes leaning, " +
  "shrink wrap stretched unevenly. The pallet shifts, tips forward and the load falls onto " +
  "the warehouse floor in realistic slow motion, leaving damaged boxes and scattered products.";

const LABEL_CLASS = "text-xs font-medium uppercase tracking-label text-sage-400";

let studioReferenceCache = null;

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const base64 = result.includes(",") ? result.split(",")[1] : result;
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.readAsDataURL(file);
  });
}

function clampDataset(value) {
  const numeric = Math.round(Number(value));
  if (Number.isNaN(numeric)) {
    return MIN_DATASET;
  }
  return Math.min(MAX_DATASET, Math.max(MIN_DATASET, numeric));
}

function batchErrorMessage(nextBatch) {
  const firstError = nextBatch.jobs?.find((job) => job.error || job.labelError || job.renderError);
  return (
    firstError?.error ||
    firstError?.labelError ||
    firstError?.renderError ||
    nextBatch.error ||
    "Generation or labeling failed."
  );
}

function compactBatchForStorage(nextBatch) {
  if (!nextBatch?.id) {
    return null;
  }
  return {
    id: nextBatch.id,
    status: nextBatch.status,
    count: nextBatch.count,
    completed: nextBatch.completed,
    failed: nextBatch.failed,
    aspect_ratio: nextBatch.aspect_ratio,
    error: nextBatch.error,
    reference: nextBatch.reference,
    jobs: (nextBatch.jobs || []).map((job) => ({
      id: job.id,
      index: job.index,
      status: job.status,
      error: job.error,
      videoUrl: job.videoUrl,
      labeledVideoUrl: job.labeledVideoUrl,
      labelStatus: job.labelStatus,
      reviewStatus: job.reviewStatus,
      labelError: job.labelError,
      renderError: job.renderError,
      cameraVariant: job.cameraVariant,
    })),
  };
}

export default function Studio() {
  const { health } = useOutletContext();
  const savedStudioState = useMemo(() => readStudioState(), []);
  const [prompt, setPrompt] = useState(savedStudioState.prompt || DEFAULT_PROMPT);
  const [aspectRatio, setAspectRatio] = useState(savedStudioState.aspectRatio || "16:9");
  const [datasetCount, setDatasetCount] = useState(() =>
    clampDataset(savedStudioState.batch?.count || DEFAULT_DATASET_COUNT),
  );
  const [reference, setReference] = useState(() => studioReferenceCache);
  const [referenceNotice, setReferenceNotice] = useState("");
  const [batch, setBatch] = useState(() => savedStudioState.batch || null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const submitTimeRef = useRef(null);
  const recordedRef = useRef(false);
  const resumeAttemptedRef = useRef(false);

  const apiKeyMissing = Boolean(health?.ok) && health.geminiApiKeyConfigured === false;
  const currentStatus = batch?.status ?? "idle";
  const isBatchActive = ["queued", "running"].includes(currentStatus);
  const canSubmit = prompt.trim().length > 0 && !isSubmitting && !isBatchActive && !apiKeyMissing;
  const busy = isSubmitting || isBatchActive;

  const safeDatasetCount = clampDataset(datasetCount || MIN_DATASET);

  const progress = useMemo(() => {
    const total = batch?.count ?? safeDatasetCount;
    const completed = batch?.completed ?? 0;
    const failed = batch?.failed ?? 0;
    const done = completed + failed;
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    return { total, completed, failed, done, pct };
  }, [batch, safeDatasetCount]);

  const previewJobs = useMemo(() => (batch?.jobs ?? []).slice(0, PREVIEW_LIMIT), [batch]);
  const downloadableVideoCount = useMemo(
    () => (batch?.jobs ?? []).filter((job) => job.videoUrl || job.labeledVideoUrl).length,
    [batch],
  );
  const placeholderCount = Math.min(PREVIEW_LIMIT, safeDatasetCount);

  useEffect(() => {
    saveStudioState({ prompt, aspectRatio, datasetCount: safeDatasetCount });
  }, [prompt, aspectRatio, safeDatasetCount]);

  useEffect(() => {
    if (batch?.id) {
      saveStudioState({ batchId: batch.id, batch: compactBatchForStorage(batch) });
    }
  }, [batch]);

  useEffect(() => {
    if (resumeAttemptedRef.current) {
      return undefined;
    }
    resumeAttemptedRef.current = true;
    const cachedBatch = savedStudioState.batch;
    const activeBatchId = savedStudioState.batchId || cachedBatch?.id || readActiveBatchId();
    if (!activeBatchId) {
      return undefined;
    }

    let cancelled = false;
    async function resumeActiveBatch() {
      try {
        const nextBatch = await getBatch(activeBatchId);
        if (cancelled) {
          return;
        }
        setBatch(nextBatch);
        if (TERMINAL_STATUSES.includes(nextBatch.status)) {
          clearActiveBatchId(nextBatch.id);
        }
        if (nextBatch.status === "failed" || nextBatch.status === "partial") {
          setError(batchErrorMessage(nextBatch));
        }
      } catch (resumeError) {
        if (!cancelled) {
          if (cachedBatch?.id === activeBatchId) {
            setBatch(cachedBatch);
          }
          setError(resumeError.message || "Could not refresh the active batch.");
        }
      }
    }

    resumeActiveBatch();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!batch?.id || TERMINAL_STATUSES.includes(batch.status)) {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextBatch = await getBatch(batch.id);
        setBatch(nextBatch);
        if (TERMINAL_STATUSES.includes(nextBatch.status)) {
          clearActiveBatchId(nextBatch.id);
        }
        if (nextBatch.status === "failed" || nextBatch.status === "partial") {
          setError(batchErrorMessage(nextBatch));
        }
      } catch (pollError) {
        setError(pollError.message || "Could not fetch status.");
      }
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [batch?.id, batch?.status]);

  useEffect(() => {
    if (!batch || !TERMINAL_STATUSES.includes(batch.status) || recordedRef.current) {
      return;
    }
    clearActiveBatchId(batch.id);
    recordedRef.current = true;
    const totalSeconds = submitTimeRef.current
      ? Math.round((Date.now() - submitTimeRef.current) / 1000)
      : null;
    recordRuns(
      (batch.jobs || []).map((job) => ({
        id: job.id,
        createdAt: new Date().toISOString(),
        status: job.status,
        labelStatus: job.labelStatus,
        reviewStatus: job.reviewStatus,
        cameraVariant: job.cameraVariant?.name,
        aspectRatio: batch.aspect_ratio,
        zoneCount: annotationCount(job.label),
        totalSeconds,
      })),
    );
  }, [batch]);

  function releaseReference() {
    if (reference?.previewUrl) {
      URL.revokeObjectURL(reference.previewUrl);
    }
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    setReferenceNotice("");
    if (!file) {
      return;
    }
    releaseReference();

    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");
    if (!isImage && !isVideo) {
      studioReferenceCache = null;
      setReference(null);
      setReferenceNotice("Unsupported file type. Please select an image or a video.");
      return;
    }
    if (isImage && file.size > MAX_IMAGE_BYTES) {
      studioReferenceCache = null;
      setReference(null);
      setReferenceNotice("Image is too large (max 8 MB).");
      return;
    }
    if (isVideo && file.size > MAX_VIDEO_BYTES) {
      studioReferenceCache = null;
      setReference(null);
      setReferenceNotice("Video is too large (max 50 MB).");
      return;
    }

    try {
      const base64 = await fileToBase64(file);
      const nextReference = {
        kind: isVideo ? "video" : "image",
        name: file.name,
        mimeType: file.type,
        data: base64,
        previewUrl: URL.createObjectURL(file),
      };
      studioReferenceCache = nextReference;
      setReference(nextReference);
    } catch (readError) {
      studioReferenceCache = null;
      setReference(null);
      setReferenceNotice(readError.message);
    }
  }

  function clearReference() {
    releaseReference();
    studioReferenceCache = null;
    setReference(null);
    setReferenceNotice("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    setBatch(null);
    recordedRef.current = false;
    submitTimeRef.current = Date.now();

    try {
      const nextBatch = await createBatch({
        prompt: prompt.trim(),
        aspectRatio,
        count: safeDatasetCount,
        reference,
      });
      saveActiveBatchId(nextBatch.id);
      saveStudioState({
        prompt: prompt.trim(),
        aspectRatio,
        datasetCount: safeDatasetCount,
        batchId: nextBatch.id,
        batch: compactBatchForStorage(nextBatch),
      });
      setBatch(nextBatch);
    } catch (submitError) {
      setError(submitError.message || "Could not start generation.");
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleDownloadAll() {
    if (!batch?.id || downloadableVideoCount === 0) {
      return;
    }
    window.location.href = getBatchDownloadUrl(batch.id);
  }

  return (
    <>
      <PageHeader
        title="Studio"
        subtitle="One prompt in, an annotated training dataset out. Describe an industrial incident and generate as many labeled videos as you need."
      />

      {apiKeyMissing ? (
        <div className="mb-6 flex items-start gap-3 rounded-md border border-accent-500/30 bg-accent-500/5 px-4 py-3 text-sm text-accent-200">
          <AlertTriangle className="mt-0.5 shrink-0 text-accent-300" size={18} aria-hidden="true" />
          <p>
            Missing Gemini API key. Add `GEMINI_API_KEY=your_key_here` to the project root `.env`
            file, then restart the backend.
          </p>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[400px_1fr]">
        <form
          onSubmit={handleSubmit}
          className="flex h-fit flex-col gap-6 rounded-lg border border-surface-700 bg-surface-900 p-5"
        >
          <label className="flex flex-col gap-2">
            <span className={LABEL_CLASS}>Prompt</span>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              maxLength={4000}
              className="min-h-44 resize-y rounded-md border border-surface-600 bg-surface-950 px-3 py-3 text-sm leading-6 text-sage-50 outline-none transition placeholder:text-sage-500 focus:border-sage-400 focus:ring-2 focus:ring-white/10"
              placeholder="Describe the industrial incident to generate..."
            />
          </label>

          <div className="flex flex-col gap-2">
            <span className={LABEL_CLASS}>Reference (optional)</span>
            {reference ? (
              <div className="flex items-center gap-3 rounded-md border border-surface-600 bg-surface-850 p-2.5">
                {reference.kind === "video" ? (
                  <video
                    src={reference.previewUrl}
                    className="h-12 w-12 rounded-md bg-black object-cover"
                    muted
                    playsInline
                  />
                ) : (
                  <img src={reference.previewUrl} alt="Reference preview" className="h-12 w-12 rounded-md object-cover" />
                )}
                <span className="min-w-0 flex-1 truncate text-xs text-sage-300">{reference.name}</span>
                <button
                  type="button"
                  onClick={clearReference}
                  className="rounded-md p-1.5 text-sage-400 transition hover:bg-surface-800 hover:text-white"
                  aria-label="Remove reference"
                >
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center justify-center gap-2 rounded-md border border-dashed border-surface-600 bg-surface-950 px-3 py-4 text-xs font-medium text-sage-400 transition hover:border-sage-400 hover:text-white"
              >
                <ImagePlus size={16} aria-hidden="true" />
                Add an image or video reference
              </button>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleFileChange}
              className="hidden"
            />
            {referenceNotice ? <p className="text-xs text-accent-300">{referenceNotice}</p> : null}
          </div>

          <div className="flex flex-col gap-3">
            <span className={LABEL_CLASS}>Format</span>
            <div className="flex gap-2">
              {["16:9", "9:16"].map((ratio) => (
                <button
                  key={ratio}
                  type="button"
                  disabled={busy}
                  onClick={() => setAspectRatio(ratio)}
                  className={`rounded-md border px-4 py-2 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${
                    aspectRatio === ratio
                      ? "border-sage-300 bg-surface-800 text-white"
                      : "border-surface-600 bg-surface-950 text-sage-400 hover:border-sage-500 hover:text-white"
                  }`}
                >
                  {ratio}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className={LABEL_CLASS}>Dataset size</span>
              <span className="text-xs font-medium text-sage-500">videos</span>
            </div>
            <input
              type="number"
              min={MIN_DATASET}
              max={MAX_DATASET}
              value={datasetCount}
              disabled={busy}
              onChange={(event) => setDatasetCount(event.target.value === "" ? "" : clampDataset(event.target.value))}
              onBlur={() => setDatasetCount((value) => clampDataset(value || MIN_DATASET))}
              className="h-11 rounded-md border border-surface-600 bg-surface-950 px-3 text-sm text-sage-50 outline-none transition disabled:cursor-not-allowed disabled:text-sage-500 focus:border-sage-400 focus:ring-2 focus:ring-white/10"
              placeholder="Number of videos to generate"
            />
          </div>

          <button
            type="submit"
            disabled={busy || !canSubmit}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-white px-4 text-sm font-medium text-black transition hover:bg-sage-200 disabled:cursor-not-allowed disabled:bg-surface-800 disabled:text-sage-500"
          >
            {apiKeyMissing ? (
              <>
                <AlertTriangle size={16} aria-hidden="true" />
                Missing API key
              </>
            ) : busy ? (
              <>
                <LoaderCircle className="animate-spin" size={16} aria-hidden="true" />
                Generating {progress.done}/{progress.total}
              </>
            ) : (
              <>
                <Play size={16} aria-hidden="true" />
                Generate dataset
              </>
            )}
          </button>

          <div className="flex items-start gap-2.5 rounded-md border border-surface-700 bg-surface-850 p-3 text-xs leading-relaxed text-sage-300">
            <Info size={15} className="mt-0.5 shrink-0 text-sage-400" aria-hidden="true" />
            <p>
              Videos share the same incident and environment with varied camera angles and details.
              Each one is automatically verified and annotated. The preview shows the first
              four; the full dataset is available for download.
            </p>
          </div>
        </form>

        <div className="flex min-h-[540px] flex-col rounded-lg border border-surface-700 bg-surface-900 p-4">
          <div className="flex items-center justify-between gap-4 border-b border-surface-700 pb-3">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-white">Output</span>
              <span className="text-xs text-sage-500">{STATUS_LABELS[currentStatus]}</span>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-3">
              <button
                type="button"
                onClick={handleDownloadAll}
                disabled={!batch?.id || downloadableVideoCount === 0}
                className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-200 transition hover:border-sage-500 hover:text-white disabled:cursor-not-allowed disabled:text-sage-500 disabled:hover:border-surface-600"
              >
                <Download size={14} aria-hidden="true" />
                Download all videos
              </button>
              <div className="h-1.5 w-40 overflow-hidden rounded-full bg-surface-700">
                <div
                  className="h-full rounded-full bg-white transition-all duration-500"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
              <span className="text-xs font-medium text-sage-300">
                {progress.done}/{progress.total}
              </span>
            </div>
          </div>

          <div className="flex flex-1 py-4">
            {previewJobs.length ? (
              <div className="grid w-full auto-rows-min gap-4 sm:grid-cols-2">
                {previewJobs.map((job) => (
                  <VideoCard key={job.id} job={job} aspectRatio={batch.aspect_ratio} />
                ))}
              </div>
            ) : (
              <div className="grid w-full auto-rows-min gap-4 sm:grid-cols-2">
                {Array.from({ length: placeholderCount }, (_, index) => (
                  <div
                    key={index}
                    className="flex flex-col overflow-hidden rounded-lg border border-dashed border-surface-600 bg-surface-900"
                  >
                    <div
                      className={`flex items-center justify-center ${
                        aspectRatio === "9:16" ? "aspect-[9/16] max-h-96" : "aspect-video"
                      }`}
                    >
                      <span className="text-5xl font-semibold text-surface-600">{index + 1}</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-surface-700 px-4 py-3">
                      <span className="text-xs font-medium text-sage-400">Sample {index + 1}</span>
                      <span className="text-xs text-sage-500">Waiting</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <p className="border-t border-surface-700 pt-3 text-center text-xs text-sage-500">
            {previewJobs.length
              ? `Previewing the first ${previewJobs.length} of ${progress.total} videos in this dataset.`
              : `Submit a prompt to generate a dataset of ${safeDatasetCount} videos (previewing the first ${placeholderCount}).`}
          </p>
        </div>
      </div>

      {error ? (
        <div className="mt-6 flex items-start gap-3 rounded-md border border-[#e5484d]/40 bg-[#e5484d]/10 px-4 py-3 text-sm text-[#ff6166]">
          <AlertTriangle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <p>{error}</p>
        </div>
      ) : null}

      <AgentLogsPanel />
    </>
  );
}
