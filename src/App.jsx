import { AlertTriangle, Download, Film, Grid2X2, LoaderCircle, Play, WandSparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const POLL_INTERVAL_MS = 3000;
const CAMERA_BATCH_SIZE = 4;

const statusLabels = {
  idle: "Ready",
  queued: "Queued",
  generating: "Generating",
  reviewing: "Gemini review",
  correcting: "Gemini correction",
  running: "Running",
  labeling: "Labeling",
  rendering: "Rendering labels",
  completed: "Video ready",
  failed: "Failed",
  partial: "Finished with errors",
};

const labelStatusLabels = {
  pending: "Annotation pending",
  running: "Annotating",
  completed: "Annotations ready",
  failed: "Annotation failed",
};

const reviewStatusLabels = {
  pending: "Review pending",
  running: "Reviewing",
  passed: "Review passed",
  failed: "Review failed",
};

const renderStatusLabels = {
  pending: "Label render pending",
  running: "Rendering labeled video",
  completed: "Labeled video ready",
  failed: "Labeled render failed",
  skipped: "Label render skipped",
};

const terminalStatuses = ["completed", "failed", "partial"];
const activeJobStatuses = ["queued", "generating", "reviewing", "correcting", "labeling", "rendering"];

async function readApiError(response) {
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

function clampBatchCount(value) {
  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return 1;
  }
  return Math.min(CAMERA_BATCH_SIZE, Math.max(1, numericValue));
}

function jobVideoUrl(job) {
  if (!job.videoUrl) {
    return "";
  }
  return `${job.videoUrl}?t=${job.id}`;
}

function jobLabeledVideoUrl(job) {
  if (!job.labeledVideoUrl) {
    return "";
  }
  return `${job.labeledVideoUrl}?t=${job.id}-labeled`;
}

function cleanLabelText(value) {
  if (value == null) {
    return "";
  }
  const text = String(value).trim();
  if (!text || ["none", "null"].includes(text.toLowerCase())) {
    return "";
  }
  return text;
}

function annotationFrames(label) {
  return Array.isArray(label?.frames) ? label.frames : [];
}

function annotationCount(label) {
  return annotationFrames(label).reduce((total, frame) => {
    return total + (Array.isArray(frame.annotations) ? frame.annotations.length : 0);
  }, 0);
}

function reviewIssues(review) {
  const issues = Array.isArray(review?.issues) ? review.issues : [];
  const missing = Array.isArray(review?.missing_requirements) ? review.missing_requirements : [];
  const visualNotes = Array.isArray(review?.visual_qa_notes) ? review.visual_qa_notes : [];
  return [...issues, ...missing, ...visualNotes].filter(Boolean);
}

export default function App() {
  const [prompt, setPrompt] = useState(
    "Continuous cinematic warehouse safety inspection video, single unbroken shot, 16:9. A large industrial warehouse aisle with tall metal storage racks, concrete floor, stacked cardboard boxes, shrink-wrapped pallets, forklift lane markings, and bright overhead lights. The camera starts with a clean before view: one loaded wooden pallet is slightly unstable on a lower rack, boxes leaning subtly, plastic wrap stretched unevenly, small warning details visible. Slowly push in toward the pallet, showing close details: wood grain, straps, labels, dust, scuffed floor, shadows, and tension in the wrapping. At 4 seconds, the pallet begins to shift and slide. Boxes wobble, the wooden pallet tips forward, and the load falls onto the warehouse floor in realistic slow motion. After the fall, the camera continues moving around the scene to show the after state: collapsed pallet, damaged boxes, torn wrap, scattered products, scrape marks on the floor, and the empty rack space above. Realistic physics, detailed textures, natural warehouse lighting, documentary safety training style, no injuries, no dialogue, ambient warehouse sound only.",
  );
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [batchCount] = useState(CAMERA_BATCH_SIZE);
  const [batch, setBatch] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const currentStatus = batch?.status ?? "idle";
  const isBatchActive = ["queued", "running"].includes(currentStatus);
  const canSubmit = prompt.trim().length > 0 && !isSubmitting && !isBatchActive;

  const progress = useMemo(() => {
    const total = batch?.count ?? batchCount;
    const completed = batch?.completed ?? 0;
    const failed = batch?.failed ?? 0;
    const done = completed + failed;
    return { total, completed, failed, done };
  }, [batch, batchCount]);

  useEffect(() => {
    if (!batch?.id || terminalStatuses.includes(batch.status)) {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/batches/${batch.id}`);
        if (!response.ok) {
          throw new Error(await readApiError(response));
        }
        const nextBatch = await response.json();
        setBatch(nextBatch);
        if (nextBatch.status === "failed" || nextBatch.status === "partial") {
          const firstError = nextBatch.jobs.find((job) => job.error || job.labelError || job.renderError);
          const errorMessage = firstError?.error || firstError?.labelError || firstError?.renderError;
          setError(errorMessage || "Generation or labeling failed.");
        }
      } catch (pollError) {
        setError(pollError.message || "Could not fetch status.");
      }
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [batch?.id, batch?.status]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    setBatch(null);

    try {
      const response = await fetch("/api/videos", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          aspect_ratio: aspectRatio,
          count: clampBatchCount(batchCount),
        }),
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      setBatch(await response.json());
    } catch (submitError) {
      setError(submitError.message || "Could not start generation.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen bg-stone-100 text-zinc-950">
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-zinc-950 text-white">
              <Film size={22} aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold">Roboss Video Lab</h1>
              <p className="text-sm text-zinc-600">Gemini + annotations VLM</p>
            </div>
          </div>
          <div className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 shadow-sm">
            {statusLabels[currentStatus]} - {progress.done}/{progress.total}
          </div>
        </header>

        <section className="grid flex-1 gap-5 lg:grid-cols-[minmax(320px,420px)_1fr]">
          <form
            onSubmit={handleSubmit}
            className="flex h-fit flex-col gap-5 rounded-lg border border-zinc-200 bg-white p-4 shadow-soft sm:p-5"
          >
            <label className="flex flex-col gap-2">
              <span className="text-sm font-semibold text-zinc-800">Prompt</span>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                className="min-h-56 resize-y rounded-md border border-zinc-300 bg-stone-50 px-3 py-3 text-base leading-6 outline-none transition focus:border-emerald-600 focus:bg-white focus:ring-4 focus:ring-emerald-100"
                maxLength={4000}
              />
            </label>

            <div className="flex flex-col gap-2">
              <span className="text-sm font-semibold text-zinc-800">Format</span>
              <div className="grid grid-cols-2 gap-2 rounded-md bg-zinc-100 p-1">
                {["16:9", "9:16"].map((ratio) => (
                  <button
                    key={ratio}
                    type="button"
                    onClick={() => setAspectRatio(ratio)}
                    className={`rounded-md px-3 py-2 text-sm font-semibold transition ${
                      aspectRatio === ratio
                        ? "bg-white text-zinc-950 shadow-sm"
                        : "text-zinc-600 hover:bg-white/70 hover:text-zinc-950"
                    }`}
                  >
                    {ratio}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-md bg-zinc-100 p-3 text-sm text-zinc-700">
              <div className="font-semibold text-zinc-900">Four-angle batch</div>
              <div className="mt-1">
                Four separate videos keep the same incident while changing only the camera angle.
              </div>
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="inline-flex h-12 items-center justify-center gap-2 rounded-md bg-emerald-700 px-4 text-base font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-500"
            >
              {isSubmitting || isBatchActive ? (
                <LoaderCircle className="animate-spin" size={20} aria-hidden="true" />
              ) : (
                <Play size={20} aria-hidden="true" />
              )}
              Generate angles
            </button>

            <div className="rounded-md bg-zinc-100 p-3">
              <div className="mb-2 flex items-center justify-between text-sm font-medium text-zinc-700">
                <span>Status</span>
                <span>{statusLabels[currentStatus]}</span>
              </div>
              <div className="mt-2 text-xs text-zinc-600">
                {progress.completed} labeled videos - {progress.failed} errors
              </div>
            </div>
          </form>

          <div className="flex min-h-[540px] flex-col rounded-lg border border-zinc-200 bg-zinc-950 p-3 shadow-soft sm:p-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3">
              <div className="flex items-center gap-2 text-white">
                <Grid2X2 size={20} aria-hidden="true" />
                <span className="font-semibold">Video preview</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-zinc-300">
                <WandSparkles size={16} aria-hidden="true" />
                <span>{statusLabels[currentStatus]}</span>
              </div>
            </div>

            <div className="flex flex-1 py-4">
              {batch?.jobs?.length ? (
                <div className="grid w-full auto-rows-fr gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  {batch.jobs.map((job) => {
                    const videoUrl = jobVideoUrl(job);
                    const labeledVideoUrl = jobLabeledVideoUrl(job);
                    const displayVideoUrl = labeledVideoUrl || videoUrl;
                    const frames = annotationFrames(job.label);
                    const zones = annotationCount(job.label);
                    const summary = cleanLabelText(job.label?.video_summary || job.label?.summary);
                    const qualityIssues = reviewIssues(job.review);
                    const reviewText = cleanLabelText(job.review?.feedback || job.review?.summary);
                    return (
                      <article
                        key={job.id}
                        className="flex min-h-72 flex-col overflow-hidden rounded-md border border-white/10 bg-white/[0.04]"
                      >
                        <div
                          className={`flex flex-1 items-center justify-center bg-black ${
                            aspectRatio === "9:16" ? "aspect-[9/16]" : "aspect-video"
                          }`}
                        >
                          {displayVideoUrl ? (
                            <video
                              key={displayVideoUrl}
                              src={displayVideoUrl}
                              controls
                              className="h-full w-full bg-black object-contain"
                            />
                          ) : (
                            <div className="flex flex-col items-center gap-3 px-4 text-center text-zinc-300">
                              {activeJobStatuses.includes(job.status) ? (
                                <LoaderCircle
                                  className="animate-spin text-emerald-300"
                                  size={28}
                                  aria-hidden="true"
                                />
                              ) : (
                                <Film size={28} aria-hidden="true" />
                              )}
                              <span className="text-sm font-semibold">{statusLabels[job.status]}</span>
                            </div>
                          )}
                        </div>
                        <div className="flex min-h-24 flex-col gap-2 p-3">
                          <div className="flex items-center justify-between gap-3">
                            <span className="text-sm font-semibold text-white">
                              Video {job.index}
                              {job.cameraVariant?.title ? ` - ${job.cameraVariant.title}` : ""}
                            </span>
                            <span className="rounded-md bg-white/10 px-2 py-1 text-xs font-medium text-zinc-200">
                              {statusLabels[job.status]}
                            </span>
                          </div>
                          {job.error ? <p className="line-clamp-2 text-xs text-red-200">{job.error}</p> : null}
                          {job.review || job.reviewStatus !== "pending" ? (
                            <div className="rounded-md bg-sky-300/10 p-2 text-xs text-sky-50">
                              <div className="flex items-center justify-between gap-2">
                                <p className="font-semibold">Gemini review</p>
                                <span className="rounded bg-sky-200/15 px-2 py-1 text-sky-100">
                                  {reviewStatusLabels[job.reviewStatus] || "Pending"}
                                </span>
                              </div>
                              {reviewText ? (
                                <p className="mt-2 text-sky-100/90">{reviewText}</p>
                              ) : null}
                              {qualityIssues.length ? (
                                <ul className="mt-2 list-disc space-y-1 pl-4 text-sky-100/80">
                                  {qualityIssues.slice(0, 4).map((issue) => (
                                    <li key={issue}>{issue}</li>
                                  ))}
                                </ul>
                              ) : null}
                              {cleanLabelText(job.correctionPrompt) && job.reviewStatus === "failed" ? (
                                <p className="mt-2 line-clamp-3 rounded bg-black/20 p-2 text-sky-100/80">
                                  Suggested correction:{" "}
                                  {cleanLabelText(job.correctionPrompt)}
                                </p>
                              ) : null}
                            </div>
                          ) : null}
                          {job.label ? (
                            <div className="rounded-md bg-white/10 p-2 text-xs text-zinc-100">
                              <div className="flex items-center justify-between gap-2">
                                <p className="font-semibold">Detected zones</p>
                                <span className="rounded bg-emerald-300/15 px-2 py-1 text-emerald-100">
                                  {zones} zones
                                </span>
                              </div>
                              {summary ? <p className="mt-1 text-zinc-300">{summary}</p> : null}
                              {Array.isArray(job.label.labels) && job.label.labels.length ? (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {job.label.labels.slice(0, 8).map((tag) => (
                                    <span key={tag} className="rounded bg-white/10 px-2 py-1 text-zinc-200">
                                      {tag}
                                    </span>
                                  ))}
                                </div>
                              ) : null}
                              {!frames.length ? (
                                <p className="mt-2 text-zinc-400">No structured zones in this label.</p>
                              ) : null}
                            </div>
                          ) : videoUrl ? (
                            <p className="text-xs text-zinc-400">
                              {labelStatusLabels[job.labelStatus] || "Annotation pending"}
                            </p>
                          ) : null}
                          {job.labelError ? (
                            <p className="line-clamp-2 text-xs text-red-200">{job.labelError}</p>
                          ) : null}
                          {videoUrl ? (
                            <div className="mt-auto grid gap-2">
                              <div className="rounded-md bg-white/10 px-3 py-2 text-xs text-zinc-200">
                                {renderStatusLabels[job.renderStatus] || "Label render pending"}
                                {job.renderError ? ` - ${job.renderError}` : ""}
                              </div>
                              <a
                                href={displayVideoUrl}
                                download
                                className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-white px-3 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-100"
                              >
                                <Download size={16} aria-hidden="true" />
                                MP4
                              </a>
                            </div>
                          ) : null}
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="flex w-full items-center justify-center">
                  <div className="flex w-full max-w-md flex-col items-center gap-4 text-center text-zinc-300">
                    <div className="flex h-16 w-16 items-center justify-center rounded-md border border-white/10 bg-white/5">
                      <Film size={30} aria-hidden="true" />
                    </div>
                    <p className="text-lg font-semibold text-white">Ready for four angles</p>
                    <p className="text-sm text-zinc-400">
                      Submit one prompt; each angle is reviewed by Gemini before labeling starts.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {error ? (
          <div className="flex items-start gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
            <AlertTriangle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
            <p>{error}</p>
          </div>
        ) : null}
      </div>
    </main>
  );
}
