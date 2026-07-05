import { useState } from "react";
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
  // Default to the labeled (annotated) video so the boxes/pose/labels show — and so the
  // player's own download control grabs the labeled file rather than the raw one.
  const [showLabeled, setShowLabeled] = useState(true);
  const preferLabeled = showLabeled && Boolean(labeledVideoUrl);
  const displayVideoUrl = preferLabeled ? labeledVideoUrl : (videoUrl || labeledVideoUrl);
  const hasBothVersions = Boolean(videoUrl) && Boolean(labeledVideoUrl);
  const isActive = ACTIVE_JOB_STATUSES.includes(job.status);
  const showStatusBadge = job.status !== "completed";

  return (
    <article className="flex flex-col overflow-hidden rounded-lg border border-surface-700 bg-surface-900 transition hover:border-surface-600">
      <div
        className={`group/video relative flex items-center justify-center bg-black ${
          aspectRatio === "9:16" ? "aspect-[9/16] max-h-96" : "aspect-video"
        } ${job.reviewStatus === "rejected" || job.status === "failed" ? "overflow-hidden" : ""}`}
      >
        {displayVideoUrl ? (
          <>
            <video
              key={displayVideoUrl}
              src={displayVideoUrl}
              controls
              preload="metadata"
              playsInline
              className={`h-full w-full bg-black object-contain ${
                job.reviewStatus === "rejected" || job.status === "failed"
                  ? "grayscale contrast-125 sepia-[0.3] hue-rotate-[-50deg] saturate-200"
                  : ""
              }`}
            />
            {hasBothVersions ? (
              <div className="absolute left-2 top-2 z-20 flex items-center gap-0.5 rounded-md bg-black/70 p-0.5 backdrop-blur-sm">
                <button
                  type="button"
                  onClick={() => setShowLabeled(true)}
                  className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide transition ${
                    preferLabeled ? "bg-neon-magenta text-[#0b0714]" : "text-sage-300 hover:text-white"
                  }`}
                >
                  Labeled
                </button>
                <button
                  type="button"
                  onClick={() => setShowLabeled(false)}
                  className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide transition ${
                    !preferLabeled ? "bg-surface-600 text-white" : "text-sage-300 hover:text-white"
                  }`}
                >
                  Raw
                </button>
              </div>
            ) : null}
            {job.reviewStatus === "rejected" || job.status === "failed" ? (
              <>
                {/* Glitch Overlay */}
                <div className="pointer-events-none absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDUiLz4KPC9zdmc+')] mix-blend-overlay opacity-50" />
                
                {/* Rejected Stamp */}
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <div className="rounded-md border-4 border-[#ff3b6b] px-4 py-2 text-3xl font-black tracking-widest text-[#ff3b6b] shadow-[0_0_20px_rgba(255,59,107,0.5)] rotate-[-12deg] bg-black/40 backdrop-blur-sm">
                    REJECTED
                  </div>
                </div>

                {/* Hover Reason Panel */}
                <div className="absolute inset-x-0 bottom-0 translate-y-full flex-col bg-black/90 border-t border-[#ff3b6b]/50 p-4 transition-transform duration-300 ease-out group-hover/video:translate-y-0 flex backdrop-blur-md">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-[#ff3b6b] uppercase tracking-wider">Physics Violation</span>
                    <span className="text-xs font-mono text-white/70 bg-white/10 px-2 py-0.5 rounded">Score: {Math.round((job.review?.score ?? 0) * 100)}/100</span>
                  </div>
                  <ul className="space-y-1.5 text-xs text-sage-200 font-mono">
                    {(job.review?.issues?.length ? job.review.issues : [job.review?.summary || job.error || "Unspecified error"]).map((violation, i) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-[#ff3b6b] mt-0.5">►</span>
                        <span className="leading-relaxed">{violation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            ) : null}
          </>
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

      <div className="flex flex-1 flex-col gap-2.5 p-4 z-10 bg-surface-900">
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
