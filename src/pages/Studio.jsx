import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Download, ImagePlus, Info, LoaderCircle, Play, X } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import PipelineCircuitPanel from "../components/PipelineCircuitPanel.jsx";
import VideoCard, { STATUS_LABELS, annotationCount } from "../components/VideoCard.jsx";
import {
  clearActiveBatchId,
  createBatch,
  createCacheDemoBatch,
  createDogDemoBatch,
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
const DOG_DEMO_PROMPT =
  "A dog moves and runs forward with a clear quadruped gait; convert that dog motion into synthetic training data for a robot dog.";
const CACHE_DEMO_PROMPT = `Realistic 4K footage, eye-level static shot, a Unitree Go2 quadruped robot dog
(appearance strictly as shown in the reference photos), on a smooth concrete
floor in a bright modern lab, soft daylight, shallow depth of field, no humans
in frame, the robot stays fully visible in frame, no cuts.

Anatomy anchor, identical throughout: the HEAD is the unit at the FRONT bearing
a black vertical face stripe with a camera lens and a round spotlight, a sensor
cluster under the chin, and the number 02 on its sides; the REAR is plain and
featureless with no camera, no lights, no markings. The robot always moves
head-first when walking forward. Never swap or mirror the head and the rear,
never show a camera on both ends, the body never reverses orientation between
consecutive moments.

Mechanical constraint: the robot's head is rigidly fixed to the body - it never
tilts, pans, nods or moves independently of the torso; only the four legs
articulate, and the whole body moves as one rigid unit.

One single continuous 30-second sequence, in this exact order:

1 (0-6 s, awakening): the robot starts lying flat on its belly with legs folded,
head pointing toward the camera, then smoothly extends all four legs and rises
into a tall standing posture, settling into active balance with tiny
micro-adjustments of its legs.

2 (6-11 s, greeting): standing, head facing the camera, it sits back slightly
onto its hind legs and raises one front leg next to its head, waving it in a
friendly greeting gesture, then returns to standing.

3 (11-19 s, exploration): it walks forward head-first with a natural dog-like
gait, then pivots 90 degrees clockwise in place SLOWLY and smoothly over about
3 full seconds - one single continuous rotation, no jumps, every intermediate
angle visible - until the head points to the right of the frame, then slows
into a careful one-leg-at-a-time walk, still head-first.

4 (19-24 s, stretch): now in side profile with the head on the right of the
frame, it bows deeply into a play-bow stretch: BOTH front legs fully extended
forward and flat on the ground, chest lowered between them, while the
featureless back half stays raised on the hind legs, then it rises back to
standing.

5 (24-30 s, calm ending): still in side profile, it lowers its back half into a
dog-like sitting pose (head up), pauses for a moment, then pushes back up with
its hind legs and returns to a tall, stable standing posture, ending motionless
and upright.`;

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

function isMissingBatchError(error) {
  return String(error?.message || "").toLowerCase().includes("batch not found");
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
  const [pipelineMode, setPipelineMode] = useState(savedStudioState.pipelineMode || "classic");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const submitTimeRef = useRef(null);
  const recordedRef = useRef(false);
  const resumeAttemptedRef = useRef(false);
  const [activeTab, setActiveTab] = useState("accepted");

  const apiKeyMissing = Boolean(health?.ok) && health.geminiApiKeyConfigured === false;
  const currentStatus = batch?.status ?? "idle";
  const isBatchActive = ["queued", "running"].includes(currentStatus);
  const isDogDemoMode = pipelineMode === "dog";
  const isCacheDemoMode = pipelineMode === "cache";
  const isDemoMode = isDogDemoMode || isCacheDemoMode;
  const displayedPrompt = isDogDemoMode ? DOG_DEMO_PROMPT : isCacheDemoMode ? CACHE_DEMO_PROMPT : prompt;
  const canSubmit =
    (isDemoMode || prompt.trim().length > 0) &&
    !isSubmitting &&
    !isBatchActive &&
    (!apiKeyMissing || isDemoMode);
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

  const acceptedJobs = useMemo(
    () =>
      (batch?.jobs ?? []).filter(
        (job) => job.status !== "failed" && !["failed", "rejected"].includes(job.reviewStatus),
      ),
    [batch]
  );

  const quarantinedJobs = useMemo(
    () =>
      (batch?.jobs ?? []).filter(
        (job) => job.status === "failed" || ["failed", "rejected"].includes(job.reviewStatus),
      ),
    [batch]
  );

  const previewJobs = useMemo(() => {
    const list = activeTab === "accepted" ? acceptedJobs : quarantinedJobs;
    return list.slice(0, PREVIEW_LIMIT);
  }, [activeTab, acceptedJobs, quarantinedJobs]);

  const downloadableVideoCount = useMemo(
    () => (batch?.jobs ?? []).filter((job) => job.videoUrl || job.labeledVideoUrl).length,
    [batch],
  );
  
  const placeholderCount = Math.min(
    PREVIEW_LIMIT,
    activeTab === "accepted" ? Math.max(0, safeDatasetCount - quarantinedJobs.length - acceptedJobs.length) : 0
  );

  useEffect(() => {
    saveStudioState({ prompt, aspectRatio, datasetCount: safeDatasetCount, pipelineMode });
  }, [prompt, aspectRatio, safeDatasetCount, pipelineMode]);

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
          if (isMissingBatchError(resumeError)) {
            clearActiveBatchId(activeBatchId);
            saveStudioState({ batchId: null, batch: null });
            setBatch(null);
            setError("");
            return;
          }
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
        if (isMissingBatchError(pollError)) {
          clearActiveBatchId(batch.id);
          saveStudioState({ batchId: null, batch: null });
          setBatch(null);
          setError("");
          return;
        }
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
      const nextBatch = isDogDemoMode
        ? await createDogDemoBatch()
        : isCacheDemoMode
          ? await createCacheDemoBatch()
          : await createBatch({
              prompt: prompt.trim(),
              aspectRatio,
              count: safeDatasetCount,
              reference,
            });
      saveActiveBatchId(nextBatch.id);
      saveStudioState({
        prompt: isDemoMode ? displayedPrompt : prompt.trim(),
        aspectRatio,
        datasetCount: isCacheDemoMode ? nextBatch.count : safeDatasetCount,
        pipelineMode,
        batchId: nextBatch.id,
        batch: compactBatchForStorage(nextBatch),
      });
      setBatch(nextBatch);
    } catch (submitError) {
      setError(submitError.message || "Could not start pipeline.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handlePipelineModeChange(nextMode) {
    setPipelineMode(nextMode);
    setError("");
    setBatch(null);
    clearActiveBatchId();
    saveStudioState({ pipelineMode: nextMode, batchId: null, batch: null });

    if (nextMode !== "cache") {
      return;
    }

    setIsSubmitting(true);
    recordedRef.current = false;
    submitTimeRef.current = Date.now();
    try {
      const nextBatch = await createCacheDemoBatch();
      saveActiveBatchId(nextBatch.id);
      saveStudioState({
        prompt: CACHE_DEMO_PROMPT,
        aspectRatio,
        datasetCount: nextBatch.count,
        pipelineMode: nextMode,
        batchId: nextBatch.id,
        batch: compactBatchForStorage(nextBatch),
      });
      setBatch(nextBatch);
      setActiveTab("accepted");
    } catch (cacheError) {
      setError(cacheError.message || "Could not load cached dataset.");
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

      {apiKeyMissing && !isDemoMode ? (
        <div className="mb-6 flex items-start gap-3 rounded-md border border-accent-500/30 bg-accent-500/5 px-4 py-3 text-sm text-accent-200">
          <AlertTriangle className="mt-0.5 shrink-0 text-accent-300" size={18} aria-hidden="true" />
          <p className="min-w-0 break-words">
            Missing Gemini API key. Add `GEMINI_API_KEY=your_key_here` to the project root `.env`
            file, then restart the backend.
          </p>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[400px_1fr]">
        <form
          onSubmit={handleSubmit}
          className="flex h-fit flex-col gap-5 rounded-lg border border-surface-700 bg-surface-900 p-4 sm:gap-6 sm:p-5"
        >
          <label className="flex flex-col gap-2">
            <span className={LABEL_CLASS}>Prompt</span>
            <textarea
              value={displayedPrompt}
              onChange={(event) => setPrompt(event.target.value)}
              disabled={isDemoMode || busy}
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
                disabled={busy || isDemoMode}
                className="flex w-full items-center justify-center gap-2 rounded-md border border-dashed border-surface-600 bg-surface-950 px-3 py-4 text-xs font-medium text-sage-400 transition hover:border-sage-400 hover:text-white disabled:cursor-not-allowed disabled:opacity-45"
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
            <div className="grid grid-cols-2 gap-2 sm:flex">
              {["16:9", "9:16"].map((ratio) => (
                <button
                  key={ratio}
                  type="button"
                  disabled={busy || isDemoMode}
                  onClick={() => setAspectRatio(ratio)}
                  className={`inline-flex justify-center rounded-md border px-4 py-2 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${
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
              disabled={busy || isDemoMode}
              onChange={(event) => setDatasetCount(event.target.value === "" ? "" : clampDataset(event.target.value))}
              onBlur={() => setDatasetCount((value) => clampDataset(value || MIN_DATASET))}
              className="h-11 rounded-md border border-surface-600 bg-surface-950 px-3 text-sm text-sage-50 outline-none transition disabled:cursor-not-allowed disabled:text-sage-500 focus:border-sage-400 focus:ring-2 focus:ring-white/10"
              placeholder="Number of videos to generate"
            />
          </div>

          <button
            type="submit"
            disabled={busy || !canSubmit}
            className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-4 text-sm font-medium text-[#0b0714] shadow-[0_0_20px_rgba(241,61,245,0.4)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-surface-800 disabled:bg-none disabled:text-sage-500 disabled:shadow-none"
          >
            {apiKeyMissing && !isDemoMode ? (
              <>
                <AlertTriangle size={16} aria-hidden="true" />
                Missing API key
              </>
            ) : busy ? (
              <>
                <LoaderCircle className="animate-spin" size={16} aria-hidden="true" />
                {isDogDemoMode
                  ? "Running dog demo"
                  : isCacheDemoMode
                    ? "Loading cache"
                    : `Generating ${progress.done}/${progress.total}`}
              </>
            ) : (
              <>
                <Play size={16} aria-hidden="true" />
                {isDogDemoMode ? "Run robot dog demo" : isCacheDemoMode ? "Load from cache" : "Generate dataset"}
              </>
            )}
          </button>

          <div className="flex items-start gap-2.5 rounded-md border border-surface-700 bg-surface-850 p-3 text-xs leading-relaxed text-sage-300">
            <Info size={15} className="mt-0.5 shrink-0 text-sage-400" aria-hidden="true" />
            <p>
              {isDogDemoMode
                ? "Dog demo mode runs a parallel front pipeline using the local ai_dog.mp4 asset. It skips generation and returns synthetic robot-dog labels/data for GO2."
                : isCacheDemoMode
                  ? "From cache mode loads generated/d79e3fa0e9ce with all cached raw videos, labeled videos, verifier reports, accepted outputs, and rejected outputs."
                  : "Videos share the same incident and environment with varied camera angles and details. Each one is automatically verified and annotated. The preview shows the first four; the full dataset is available for download."}
            </p>
          </div>
        </form>

        <div className="flex min-h-[420px] flex-col rounded-lg border border-surface-700 bg-surface-900 p-3 sm:min-h-[540px] sm:p-4">
          <div className="flex flex-col items-stretch justify-between gap-3 border-b border-surface-700 pb-3 sm:flex-row sm:items-center sm:gap-4">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-white">Output</span>
                <span className="text-xs text-sage-500">{STATUS_LABELS[currentStatus]}</span>
              </div>

              <div className="flex items-center gap-2 rounded-md bg-surface-950 p-1">
                <button
                  type="button"
                  onClick={() => setActiveTab("accepted")}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                    activeTab === "accepted"
                      ? "bg-surface-800 text-neon-green"
                      : "text-sage-500 hover:text-white"
                  }`}
                >
                  Accepted ({acceptedJobs.length})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("quarantined")}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                    activeTab === "quarantined"
                      ? "bg-surface-800 text-[#ff3b6b]"
                      : "text-sage-500 hover:text-white"
                  }`}
                >
                  Quarantined ({quarantinedJobs.length})
                </button>
              </div>
            </div>

            <div className="flex w-full flex-wrap items-center justify-end gap-3 sm:w-auto">
              <div className="flex items-center gap-2 rounded-md border border-surface-700 bg-surface-850 px-3 py-1.5 hidden lg:flex">
                 <span className="text-xs font-mono text-sage-400">Gen: {progress.done}</span>
                 <span className="text-sage-600">→</span>
                 <span className="text-xs font-mono text-[#ff3b6b]">Rej: {quarantinedJobs.length}</span>
                 <span className="text-sage-600">→</span>
                 <span className="text-xs font-mono text-neon-green">Yield: {progress.done > 0 ? Math.round((acceptedJobs.length / progress.done) * 100) : 0}%</span>
              </div>

              <button
                type="button"
                onClick={handleDownloadAll}
                disabled={!batch?.id || downloadableVideoCount === 0}
                className="inline-flex h-9 w-full items-center justify-center gap-2 rounded-md border border-surface-600 bg-surface-950 px-3 text-xs font-medium text-sage-200 transition hover:border-sage-500 hover:text-white disabled:cursor-not-allowed disabled:text-sage-500 disabled:hover:border-surface-600 sm:w-auto"
              >
                <Download size={14} aria-hidden="true" />
                Download dataset
              </button>
              <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-700 sm:w-40 sm:flex-none">

                <div
                  className="h-full rounded-full bg-gradient-to-r from-neon-magenta to-neon-cyan transition-all duration-500"
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
              <div className="grid w-full auto-rows-min gap-4 md:grid-cols-2">
                {previewJobs.map((job) => (
                  <VideoCard key={job.id} job={job} aspectRatio={batch.aspect_ratio} />
                ))}
              </div>
            ) : (
              <div className="grid w-full auto-rows-min gap-4 md:grid-cols-2">
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
              : isDogDemoMode
                ? "Run the robot dog demo pipeline to preview the bundled dog-motion clip and synthetic labels."
                : isCacheDemoMode
                  ? "Load the cached generated/d79e3fa0e9ce dataset to preview accepted and rejected cached videos."
                  : `Submit a prompt to generate a dataset of ${safeDatasetCount} videos (previewing the first ${placeholderCount}).`}
          </p>
        </div>
      </div>

      {error ? (
        <div className="mt-6 flex items-start gap-3 rounded-md border border-[#ff3b6b]/40 bg-[#ff3b6b]/10 px-4 py-3 text-sm text-[#ff3b6b]">
          <AlertTriangle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <p className="min-w-0 break-words">{error}</p>
        </div>
      ) : null}

      <section className="mt-6 rounded-lg border border-surface-700 bg-surface-900 p-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-label text-sage-400">Pipeline mode</p>
            <h2 className="mt-1 text-sm font-medium text-white">
              Choose the real pipeline, the robot dog demo, or a cached generated run
            </h2>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <button
              type="button"
              disabled={busy}
              onClick={() => handlePipelineModeChange("classic")}
              className={`rounded-md border px-4 py-3 text-left transition disabled:cursor-not-allowed disabled:opacity-50 ${
                pipelineMode === "classic"
                  ? "border-neon-cyan/60 bg-neon-cyan/10 text-white shadow-[0_0_18px_rgba(47,232,234,0.18)]"
                  : "border-surface-600 bg-surface-950 text-sage-300 hover:border-sage-500 hover:text-white"
              }`}
            >
              <span className="block text-xs font-semibold uppercase tracking-label">Real pipeline</span>
              <span className="mt-1 block text-xs text-sage-400">Prompt → generate → verify → labels</span>
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => handlePipelineModeChange("dog")}
              className={`rounded-md border px-4 py-3 text-left transition disabled:cursor-not-allowed disabled:opacity-50 ${
                isDogDemoMode
                  ? "border-neon-violet/70 bg-neon-violet/10 text-white shadow-[0_0_18px_rgba(139,92,246,0.22)]"
                  : "border-surface-600 bg-surface-950 text-sage-300 hover:border-sage-500 hover:text-white"
              }`}
            >
              <span className="block text-xs font-semibold uppercase tracking-label">Robot dog demo</span>
              <span className="mt-1 block text-xs text-sage-400">ai_dog.mp4 → synthetic GO2 data</span>
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => handlePipelineModeChange("cache")}
              className={`rounded-md border px-4 py-3 text-left transition disabled:cursor-not-allowed disabled:opacity-50 ${
                isCacheDemoMode
                  ? "border-neon-green/70 bg-neon-green/10 text-white shadow-[0_0_18px_rgba(126,247,144,0.2)]"
                  : "border-surface-600 bg-surface-950 text-sage-300 hover:border-sage-500 hover:text-white"
              }`}
            >
              <span className="block text-xs font-semibold uppercase tracking-label">From cache</span>
              <span className="mt-1 block text-xs text-sage-400">generated/d79e3fa0e9ce → full dataset</span>
            </button>
          </div>
        </div>
      </section>

      <PipelineCircuitPanel batch={batch} />
    </>
  );
}
