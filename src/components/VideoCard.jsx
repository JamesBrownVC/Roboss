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
  completed: "Completed",
  failed: "Failed",
  partial: "Finished with errors",
};

const ACTIVE_JOB_STATUSES = ["queued", "generating", "reviewing", "correcting", "labeling", "rendering"];

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
  const isActive = ACTIVE_JOB_STATUSES.includes(job.status);
  const showStatusBadge = job.status !== "completed";

  return (
    <article className="flex flex-col overflow-hidden rounded-lg border border-surface-700 bg-surface-900 transition hover:border-surface-600">
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
              <Film className="text-sage-500" size={28} aria-hidden="true" />
            )}
            <span className="text-sm font-medium text-sage-300">
              {STATUS_LABELS[job.status] || job.status}
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-2.5 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="min-w-0 text-sm font-semibold text-white">
            {job.cameraVariant?.title || `Video ${job.index}`}
          </span>
          {showStatusBadge ? (
            <span
              className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${
                job.status === "failed"
                  ? "border-[#ff3b6b]/40 bg-[#ff3b6b]/10 text-[#ff3b6b]"
                  : "border-surface-600 bg-surface-850 text-sage-300"
              }`}
            >
              {STATUS_LABELS[job.status] || job.status}
            </span>
          ) : null}
        </div>

        {job.error ? <p className="line-clamp-2 text-xs text-[#ff3b6b]">{job.error}</p> : null}

        {job.labelError ? <p className="line-clamp-2 text-xs text-[#ff3b6b]">{job.labelError}</p> : null}

        {videoUrl ? (
          <div className={`mt-auto grid gap-2 pt-1 ${labeledVideoUrl ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1"}`}>
            <a
              href={videoUrl}
              download
              className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-surface-600 px-3 text-xs font-medium text-sage-200 transition hover:border-sage-500 hover:text-white"
            >
              <Download size={14} aria-hidden="true" />
              Raw MP4
            </a>
            {labeledVideoUrl ? (
              <a
                href={labeledVideoUrl}
                download
                className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-gradient-to-r from-neon-magenta to-neon-violet px-3 text-xs font-medium text-[#0b0714] shadow-[0_0_16px_rgba(241,61,245,0.35)] transition hover:brightness-110"
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
