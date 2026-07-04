import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ImagePlus, Info, LoaderCircle, Play, X } from "lucide-react";
import PageHeader from "../components/PageHeader.jsx";
import VideoCard, { STATUS_LABELS, annotationCount } from "../components/VideoCard.jsx";
import { createBatch, getBatch, recordRuns } from "../lib/api.js";

const POLL_INTERVAL_MS = 3000;
const PREVIEW_LIMIT = 4;
const MIN_DATASET = 1;
const MAX_DATASET = 5000;
const TERMINAL_STATUSES = ["completed", "failed", "partial"];
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_VIDEO_BYTES = 50 * 1024 * 1024;

const DEFAULT_PROMPT =
  "A loaded wooden pallet sits slightly unstable on a lower warehouse rack, boxes leaning, " +
  "shrink wrap stretched unevenly. The pallet shifts, tips forward and the load falls onto " +
  "the warehouse floor in realistic slow motion, leaving damaged boxes and scattered products.";

const LABEL_CLASS = "text-xs font-semibold uppercase tracking-label text-sage-300/70";

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

export default function Studio() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [datasetCount, setDatasetCount] = useState(10);
  const [reference, setReference] = useState(null);
  const [referenceNotice, setReferenceNotice] = useState("");
  const [batch, setBatch] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const submitTimeRef = useRef(null);
  const recordedRef = useRef(false);

  const currentStatus = batch?.status ?? "idle";
  const isBatchActive = ["queued", "running"].includes(currentStatus);
  const canSubmit = prompt.trim().length > 0 && !isSubmitting && !isBatchActive;
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
  const placeholderCount = Math.min(PREVIEW_LIMIT, safeDatasetCount);

  useEffect(() => {
    if (!batch?.id || TERMINAL_STATUSES.includes(batch.status)) {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextBatch = await getBatch(batch.id);
        setBatch(nextBatch);
        if (nextBatch.status === "failed" || nextBatch.status === "partial") {
          const firstError = nextBatch.jobs.find((job) => job.error || job.labelError || job.renderError);
          const message = firstError?.error || firstError?.labelError || firstError?.renderError;
          setError(message || "Generation or labeling failed.");
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
      setReference(null);
      setReferenceNotice("Unsupported file type. Please select an image or a video.");
      return;
    }
    if (isImage && file.size > MAX_IMAGE_BYTES) {
      setReference(null);
      setReferenceNotice("Image is too large (max 8 MB).");
      return;
    }
    if (isVideo && file.size > MAX_VIDEO_BYTES) {
      setReference(null);
      setReferenceNotice("Video is too large (max 50 MB).");
      return;
    }

    try {
      const base64 = await fileToBase64(file);
      setReference({
        kind: isVideo ? "video" : "image",
        name: file.name,
        mimeType: file.type,
        data: base64,
        previewUrl: URL.createObjectURL(file),
      });
    } catch (readError) {
      setReference(null);
      setReferenceNotice(readError.message);
    }
  }

  function clearReference() {
    releaseReference();
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
      setBatch(nextBatch);
    } catch (submitError) {
      setError(submitError.message || "Could not start generation.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Studio"
        subtitle="One prompt in, an annotated training dataset out. Describe an industrial incident and generate as many review-checked, labeled videos as you need."
      />

      <div className="grid gap-6 xl:grid-cols-[400px_1fr]">
        <form
          onSubmit={handleSubmit}
          className="flex h-fit flex-col gap-6 rounded-xl border border-white/5 bg-surface-900 p-5 shadow-soft"
        >
          <label className="flex flex-col gap-2">
            <span className={LABEL_CLASS}>Prompt</span>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              maxLength={4000}
              className="min-h-44 resize-y rounded-lg border border-white/10 bg-surface-850 px-3 py-3 text-sm leading-6 text-sage-50 outline-none transition placeholder:text-sage-300/30 focus:border-accent-500/60 focus:ring-4 focus:ring-accent-500/10"
              placeholder="Describe the industrial incident to generate..."
            />
          </label>

          <div className="flex flex-col gap-2">
            <span className={LABEL_CLASS}>Reference (optional)</span>
            {reference ? (
              <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-surface-850 p-2.5">
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
                <span className="min-w-0 flex-1 truncate text-xs text-sage-200/80">{reference.name}</span>
                <button
                  type="button"
                  onClick={clearReference}
                  className="rounded-md p-1.5 text-sage-300/60 transition hover:bg-white/5 hover:text-sage-100"
                  aria-label="Remove reference"
                >
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center justify-center gap-2 rounded-lg border border-dashed border-white/15 bg-surface-850 px-3 py-4 text-xs font-semibold text-sage-300/70 transition hover:border-accent-500/50 hover:text-sage-100"
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
            {referenceNotice ? <p className="text-xs text-amber-300/90">{referenceNotice}</p> : null}
          </div>

          <div className="flex flex-col gap-3">
            <span className={LABEL_CLASS}>Format</span>
            <div className="flex gap-2">
              {["16:9", "9:16"].map((ratio) => (
                <button
                  key={ratio}
                  type="button"
                  onClick={() => setAspectRatio(ratio)}
                  className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
                    aspectRatio === ratio
                      ? "bg-accent-500/10 text-accent-200 ring-1 ring-accent-500/40"
                      : "bg-surface-850 text-sage-300/60 hover:text-sage-100"
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
              <span className="text-xs font-semibold text-sage-300/60">videos</span>
            </div>
            <input
              type="number"
              min={MIN_DATASET}
              max={MAX_DATASET}
              value={datasetCount}
              onChange={(event) => setDatasetCount(event.target.value === "" ? "" : clampDataset(event.target.value))}
              onBlur={() => setDatasetCount((value) => clampDataset(value || MIN_DATASET))}
              className="h-11 rounded-lg border border-white/10 bg-surface-850 px-3 text-sm text-sage-50 outline-none transition focus:border-accent-500/60 focus:ring-4 focus:ring-accent-500/10"
              placeholder="Number of videos to generate"
            />
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-accent-500 px-4 text-base font-bold text-surface-950 shadow-glow transition hover:bg-accent-400 disabled:cursor-not-allowed disabled:bg-surface-800 disabled:text-sage-300/30 disabled:shadow-none"
          >
            {busy ? (
              <>
                <LoaderCircle className="animate-spin" size={20} aria-hidden="true" />
                Generating {progress.done}/{progress.total}
              </>
            ) : (
              <>
                <Play size={20} aria-hidden="true" />
                Generate dataset
              </>
            )}
          </button>

          <div className="flex items-start gap-2.5 rounded-lg border border-sage-400/15 bg-white/[0.02] p-3 text-xs leading-relaxed text-sage-200/80">
            <Info size={15} className="mt-0.5 shrink-0 text-sage-300" aria-hidden="true" />
            <p>
              Videos share the same incident and environment with varied camera angles and details.
              Each one is automatically quality-reviewed and annotated. The preview shows the first
              four; the full dataset is available for download.
            </p>
          </div>
        </form>

        <div className="flex min-h-[540px] flex-col rounded-xl border border-white/5 bg-surface-900 p-4 shadow-soft">
          <div className="flex items-center justify-between gap-4 border-b border-white/5 pb-3">
            <div className="flex items-center gap-3">
              <span className="font-display font-semibold text-white">Output</span>
              <span className="text-xs text-sage-300/50">{STATUS_LABELS[currentStatus]}</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-1.5 w-40 overflow-hidden rounded-full bg-surface-700">
                <div
                  className="h-full rounded-full bg-accent-500 transition-all duration-500"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-sage-200/70">
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
                    className="flex flex-col overflow-hidden rounded-xl border border-dashed border-white/10 bg-surface-850/60"
                  >
                    <div
                      className={`flex items-center justify-center ${
                        aspectRatio === "9:16" ? "aspect-[9/16] max-h-96" : "aspect-video"
                      }`}
                    >
                      <span className="font-display text-5xl font-bold text-sage-500/30">{index + 1}</span>
                    </div>
                    <div className="flex items-center justify-between px-4 py-3">
                      <span className="text-xs font-semibold text-sage-300/60">Sample {index + 1}</span>
                      <span className="text-xs text-sage-300/35">Waiting</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <p className="border-t border-white/5 pt-3 text-center text-xs text-sage-300/40">
            {previewJobs.length
              ? `Previewing the first ${previewJobs.length} of ${progress.total} videos in this dataset.`
              : `Submit a prompt to generate a dataset of ${safeDatasetCount} videos (previewing the first ${placeholderCount}).`}
          </p>
        </div>
      </div>

      {error ? (
        <div className="mt-6 flex items-start gap-3 rounded-lg border border-red-400/20 bg-red-400/5 px-4 py-3 text-sm text-red-200">
          <AlertTriangle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <p>{error}</p>
        </div>
      ) : null}
    </>
  );
}
