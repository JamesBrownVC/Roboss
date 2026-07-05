"""Render the labeling process as demo artifacts for one episode:

  demo/label_demo/{name}.mp4  - the video with SuperAnimal keypoints, the
                                active segment banner, a colored segment
                                timeline, and the agent's tool steps appearing
                                as they "happen"
  demo/label_demo/{name}.png  - storyboard: keyframes with keypoints, the
                                agent action flow (incl. critic verdicts),
                                the segment gantt and the cmd_twist profile

Usage:
  python scripts/render_label_demo.py <episode_id> [<out_name>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2r.config import V2RConfig  # noqa: E402
from v2r.schema.workspace import EpisodeWorkspace  # noqa: E402

LEG_CHAINS = {
    "FL": ["front_left_thai", "front_left_knee", "front_left_paw"],
    "FR": ["front_right_thai", "front_right_knee", "front_right_paw"],
    "RL": ["back_left_thai", "back_left_knee", "back_left_paw"],
    "RR": ["back_right_thai", "back_right_knee", "back_right_paw"],
}
SPINE = ["nose", "neck_base", "back_base", "back_middle", "back_end", "tail_base"]
LEG_COLOR = {"FL": (60, 160, 255), "FR": (40, 90, 230), "RL": (255, 170, 60), "RR": (230, 110, 40)}
SPINE_COLOR = (90, 220, 120)
SEG_PALETTE = [(96, 189, 90), (222, 155, 67), (108, 128, 235), (90, 200, 220),
               (200, 100, 200), (120, 220, 160), (240, 120, 120)]


def _load(ws: EpisodeWorkspace):
    kp = None
    kp_path = ws.root / "animal" / "keypoints_superanimal.parquet"
    if kp_path.is_file():
        kp = pd.read_parquet(kp_path)
    segments = []
    if ws.segments_json.is_file():
        segments = json.loads(ws.segments_json.read_text(encoding="utf-8"))["segments"]
    transcript = []
    tpath = ws.qa_dir / "agentic_transcript.json"
    if tpath.is_file():
        transcript = json.loads(tpath.read_text(encoding="utf-8"))["transcript"]
    report = {}
    rpath = ws.qa_dir / "agentic_label_report.json"
    if rpath.is_file():
        report = json.loads(rpath.read_text(encoding="utf-8"))
    twist = None
    tw = ws.retarget_dir("go2") / "cmd_twist.parquet"
    if tw.is_file():
        twist = pd.read_parquet(tw)
    utterances = []
    if ws.utterances_json.is_file():
        utterances = json.loads(ws.utterances_json.read_text(encoding="utf-8")).get("utterances", [])
    fit = {}
    fpath = ws.retarget_dir("go2") / "twin_fit_report.json"
    if fpath.is_file():
        fit = json.loads(fpath.read_text(encoding="utf-8"))
    return kp, segments, transcript, report, twist, fit, utterances


def _kp_frames(kp: pd.DataFrame):
    """t -> {name: (u, v, conf)} lookup, plus sorted times."""
    out = {}
    for t, grp in kp.groupby("t"):
        out[float(t)] = {r.keypoint_name: (r.u, r.v, r.conf) for r in grp.itertuples()}
    return out, np.array(sorted(out))


def _draw_kp(frame, pts: dict, w: int, h: int):
    def px(name):
        if name in pts and pts[name][2] > 0.3:
            u, v, _ = pts[name]
            return int(u * w), int(v * h)
        return None

    for chain, color in [(SPINE, SPINE_COLOR)] + [(c, LEG_COLOR[l]) for l, c in LEG_CHAINS.items()]:
        prev = None
        for name in chain:
            p = px(name)
            if p and prev:
                cv2.line(frame, prev, p, color, 2, cv2.LINE_AA)
            prev = p or prev
    for name, (u, v, c) in pts.items():
        if c > 0.3:
            cv2.circle(frame, (int(u * w), int(v * h)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(u * w), int(v * h)), 3, (30, 30, 30), 1, cv2.LINE_AA)


def _step_label(s: dict) -> str:
    a = s["action"]
    if a == "run_tool":
        return f"tool: {s['args'].get('name', '?')}"
    if a == "look":
        return f"look ({len(s['args'].get('timestamps', []) or [s['args'].get('n', '?')])} frames)"
    if a == "finalize":
        obs = str(s.get("observation", ""))
        return "finalize -> critic " + ("ACCEPT" if "accept" in obs else "REVISE" if "revise" in obs else "")
    return a


def render_video(ws, out_mp4: Path, kp, segments, transcript, report, utterances=None):
    cap = cv2.VideoCapture(str(ws.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    strip = 130
    outw, outh = W, H + strip
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (outw, outh))
    kp_lookup, kp_times = _kp_frames(kp) if kp is not None else ({}, np.array([]))
    steps = [_step_label(s) for s in transcript]
    feas = report.get("feasibility", {})

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        # keypoints (nearest sampled time within 0.2 s)
        if len(kp_times):
            j = int(np.argmin(np.abs(kp_times - t)))
            if abs(kp_times[j] - t) < 0.2:
                _draw_kp(frame, kp_lookup[kp_times[j]], W, H)

        # header
        cv2.rectangle(frame, (0, 0), (W, 34), (25, 25, 25), -1)
        head = (f"V2R agentic labeling | {ws.episode_id} | "
                f"judge: {report.get('judge_source', '?')[:46]} | "
                f"verdict: {feas.get('recommendation', '?')}")
        cv2.putText(frame, head, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        # agent step panel (steps appear over time)
        visible = min(len(steps), 1 + int(t / max(duration, 1e-6) * len(steps)))
        y0 = 46
        for k in range(visible):
            active = k == visible - 1
            col = (80, 235, 255) if active else (200, 200, 200)
            cv2.putText(frame, f"{k + 1}. {steps[k][:34]}", (W - 330, y0 + 20 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        # bottom strip: segment timeline + active label
        canvas = np.full((outh, outw, 3), 22, dtype=np.uint8)
        canvas[:H] = frame
        bar_y0, bar_y1 = H + 18, H + 52
        active_seg = None
        for si, seg in enumerate(segments):
            x0 = int(seg["start_s"] / duration * (W - 20)) + 10
            x1 = int(seg["end_s"] / duration * (W - 20)) + 10
            color = SEG_PALETTE[si % len(SEG_PALETTE)]
            cv2.rectangle(canvas, (x0, bar_y0), (x1, bar_y1), color, -1)
            cv2.putText(canvas, seg["skill"], (x0 + 4, bar_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
            if seg["start_s"] <= t <= seg["end_s"]:
                active_seg = seg
        px_t = int(t / duration * (W - 20)) + 10
        cv2.line(canvas, (px_t, bar_y0 - 6), (px_t, bar_y1 + 6), (255, 255, 255), 2)
        if active_seg:
            cv2.putText(canvas, f"{active_seg['skill'].upper()}: {active_seg['text'][:80]}",
                        (10, H + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 170), 2, cv2.LINE_AA)
        # speech subtitles from the utterance channel
        for u in utterances or []:
            if u["t_start_s"] <= t <= u["t_end_s"] + 0.3:
                sub = f'{u.get("speaker", "?")}: "{u["text"][:70]}"'
                (tw_, th_), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                x = max((W - tw_) // 2, 8)
                cv2.rectangle(canvas, (x - 8, H - 46 - th_), (x + tw_ + 8, H - 30), (15, 15, 15), -1)
                cv2.putText(canvas, sub, (x, H - 38), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (90, 220, 255), 2, cv2.LINE_AA)
                break
        cv2.putText(canvas, f"t = {t:5.2f} s", (W - 150, H + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "keypoints: SuperAnimal-Quadruped (39) | labels: Nemotron agent + Kimi critic | source: estimated",
                    (10, H + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(canvas)
        i += 1
    cap.release()
    vw.release()


def render_storyboard(ws, out_png: Path, kp, segments, transcript, report, twist, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    cap = cv2.VideoCapture(str(ws.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    kp_lookup, kp_times = _kp_frames(kp) if kp is not None else ({}, np.array([]))

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[2.1, 1.6, 1.2], hspace=0.42, wspace=0.15)

    # row 1: keyframes with keypoints
    for c, frac in enumerate([0.08, 0.28, 0.5, 0.72, 0.92]):
        ax = fig.add_subplot(gs[0, c])
        idx = int(frac * (n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        t = idx / fps
        if len(kp_times):
            j = int(np.argmin(np.abs(kp_times - t)))
            if abs(kp_times[j] - t) < 0.25:
                _draw_kp(frame, kp_lookup[kp_times[j]], frame.shape[1], frame.shape[0])
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        seg = next((s for s in segments if s["start_s"] <= t <= s["end_s"]), None)
        ax.set_title(f"t={t:.1f}s  [{seg['skill'] if seg else '-'}]", fontsize=10)
        ax.axis("off")
    cap.release()

    # row 2 left (3 cols): agent action flow
    ax = fig.add_subplot(gs[1, :3])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Agent investigation (Nemotron orchestrator, Kimi critic)", fontsize=11, loc="left")
    per_row = 4
    for k, s in enumerate(transcript):
        col, row = k % per_row, k // per_row
        x, y = 0.02 + col * 0.25, 0.78 - row * 0.34
        label = _step_label(s)
        fc = ("#2e7d32" if "ACCEPT" in label else "#c62828" if "REVISE" in label
              else "#1565c0" if label.startswith("tool") else "#455a64")
        ax.add_patch(FancyBboxPatch((x, y - 0.11), 0.21, 0.2,
                                    boxstyle="round,pad=0.012", fc=fc, ec="none", alpha=0.9))
        ax.text(x + 0.105, y, f"{k + 1}. {label[:26]}", ha="center", va="center",
                fontsize=8.2, color="white", wrap=True)
        if col < per_row - 1 and k + 1 < len(transcript):
            ax.annotate("", xy=(x + 0.245, y), xytext=(x + 0.225, y),
                        arrowprops=dict(arrowstyle="->", color="#888"))

    # row 2 right (2 cols): segment gantt
    ax = fig.add_subplot(gs[1, 3:])
    ax.set_title("Labeled segments", fontsize=11, loc="left")
    for si, seg in enumerate(segments):
        color = np.array(SEG_PALETTE[si % len(SEG_PALETTE)][::-1]) / 255
        ax.barh(0, seg["end_s"] - seg["start_s"], left=seg["start_s"], height=0.5, color=color)
        ax.text((seg["start_s"] + seg["end_s"]) / 2, 0, seg["skill"], ha="center",
                va="center", fontsize=9)
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.set_xlabel("t (s)")

    # row 3: twist profile + summary
    ax = fig.add_subplot(gs[2, :3])
    if twist is not None:
        ax.plot(twist.t, twist.vx, "-", color="#1565c0", label="vx (m/s)")
        ax.plot(twist.t, twist.yaw_rate, "-", color="#e65100", label="yaw rate (rad/s)")
        ax.fill_between(twist.t, 0, twist.conf * 0.1, color="#9e9e9e", alpha=0.4,
                        label="path confidence (x0.1)")
        ax.legend(fontsize=8, ncol=3)
        ax.set_title("Go2 command channel (cmd_twist) from the twin fit", fontsize=11, loc="left")
        ax.set_xlabel("t (s)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no twin fit for this episode", ha="center", fontsize=11)

    ax = fig.add_subplot(gs[2, 3:])
    ax.axis("off")
    feas = report.get("feasibility", {})
    lines = [
        f"episode: {ws.episode_id}",
        f"judge: {report.get('judge_source', '?')}",
        f"verdict: {feas.get('recommendation', '?')} (conf {feas.get('confidence', 0):.2f})",
        f"AI-generated: {feas.get('ai_generated_suspected', '?')}",
    ]
    if fit:
        lines += [
            f"twin gait: {fit.get('gait_label')} | stride {fit.get('stride_period_s', 0):.2f}s",
            f"tracking loss: {fit.get('loss_initial', 0):.3f} -> {fit.get('loss_final', 0):.3f} "
            f"({fit.get('loss_reduction_pct', 0)}% reduction)",
        ]
    ax.text(0.02, 0.92, "\n".join(lines), va="top", fontsize=10.5, family="monospace")

    fig.suptitle(f"V2R agentic labeling + digital-twin fit - {ws.episode_id}",
                 fontsize=14, y=0.995)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    episode = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else episode
    cfg = V2RConfig.load(Path(__file__).resolve().parents[1])
    ws = EpisodeWorkspace(cfg.workspaces_root, episode)
    out_dir = cfg.root.parent / "demo" / "label_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    kp, segments, transcript, report, twist, fit, utterances = _load(ws)
    render_video(ws, out_dir / f"{name}.mp4", kp, segments, transcript, report, utterances)
    render_storyboard(ws, out_dir / f"{name}.png", kp, segments, transcript, report, twist, fit)
    print("wrote", out_dir / f"{name}.mp4")
    print("wrote", out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
