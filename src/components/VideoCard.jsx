import { Download, Film, LoaderCircle } from "lucide-react";

export const STATUS_LABELS = {
  idle: "Ready",
  queued: "Queued",
  generating: "Generating",
  reviewing: "Verifying",
  correcting: "Correcting",
  running: "Running",
  labeling: "Building labels",
  rendering: "Rendering preview",
  completed: "Video ready",
  failed: "Failed",
  partial: "Finished with errors",
};

const ACTIVE_JOB_STATUSES = ["queued", "generating", "reviewing", "correcting", "labeling", "rendering"];

function cleanText(value) {
  if (value == null) {
    return "";
  }
  const text = String(value).trim();
  if (!text || ["none", "null"].includes(text.toLowerCase())) {
    return "";
  }
  return text;
}

export function annotationCount(label) {
  const frames = Array.isArray(label?.frames) ? label.frames : [];
  return frames.reduce(
    (total, frame) => total + (Array.isArray(frame.annotations) ? frame.annotations.length : 0),
    0,
  );
}


export default function VideoCard({ job, aspectRatio }) {
  const videoUrl = job.videoUrl ? `${job.videoUrl}?t=${job.id}` : "";
  const labeledVideoUrl = job.labeledVideoUrl ? `${job.labeledVideoUrl}?t=${job.id}-labeled` : "";
  const displayVideoUrl = videoUrl || labeledVideoUrl;
  const zones = annotationCount(job.label);
  const summary = cleanText(job.label?.video_summary || job.label?.summary);
  const isActive = ACTIVE_JOB_STATUSES.includes(job.status);

  return (
    <article className="flex flex-col overflow-hidden rounded-xl border border-white/5 bg-surface-850 shadow-soft">
      <div
        className={`flex items-center justify-center bg-black ${
          aspectRatio === "9:16" ? "aspect-[9/16] max-h-96" : "aspect-video"
        }`}
      >
        {displayVideoUrl ? (
          <video
            key={displayVideoUrl}
            src={displayVideoUrl}
            controls
            preload="metadata"
            playsInline
            className="h-full w-full bg-black object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 px-4 text-center">
            {isActive ? (
              <LoaderCircle className="animate-spin text-sage-300" size={28} aria-hidden="true" />
            ) : (
              <Film className="text-sage-300/40" size={28} aria-hidden="true" />
            )}
            <span className="text-sm font-semibold text-sage-200/70">
              {STATUS_LABELS[job.status] || job.status}
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-2.5 p-4">
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm font-semibold text-white">
            {job.cameraVariant?.title || `Video ${job.index}`}
          </span>
          <span
            className={`rounded-md px-2 py-1 text-xs font-semibold ${
              job.status === "failed"
                ? "bg-red-400/10 text-red-300"
                : job.status === "completed"
                  ? "bg-emerald-400/15 text-emerald-300"
                  : "bg-sage-500/10 text-sage-200/70"
            }`}
          >
            {STATUS_LABELS[job.status] || job.status}
          </span>
        </div>

        {job.error ? <p className="line-clamp-2 text-xs text-red-300">{job.error}</p> : null}

        {job.label ? (
          <div className="rounded-lg bg-surface-800 p-2.5 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-sage-200/80">Detected zones</span>
              <span className="rounded bg-emerald-400/15 px-2 py-0.5 font-medium text-emerald-300">
                {zones} zones
              </span>
            </div>
            {summary ? <p className="mt-1.5 text-sage-200/50">{summary}</p> : null}
            {Array.isArray(job.label.labels) && job.label.labels.length ? (
              <div className="mt-2 flex flex-wrap gap-1">
                {job.label.labels.slice(0, 8).map((tag) => (
                  <span key={tag} className="rounded bg-sage-500/10 px-2 py-0.5 text-sage-200/80">
                    {tag}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {job.labelError ? <p className="line-clamp-2 text-xs text-red-300">{job.labelError}</p> : null}

        {videoUrl ? (
          <div className={`mt-auto grid gap-2 pt-1 ${labeledVideoUrl ? "grid-cols-2" : "grid-cols-1"}`}>
            <a
              href={videoUrl}
              download
              className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-white/10 px-3 text-xs font-semibold text-sage-200 transition hover:bg-white/5"
            >
              <Download size={14} aria-hidden="true" />
              Raw MP4
            </a>
            {labeledVideoUrl ? (
              <a
                href={labeledVideoUrl}
                download
                className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-emerald-500 px-3 text-xs font-semibold text-surface-950 transition hover:bg-emerald-400"
              >
                <Download size={14} aria-hidden="true" />
                Labeled MP4
              </a>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}
