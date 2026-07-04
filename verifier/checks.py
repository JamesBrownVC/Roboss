"""Physics-aware plausibility checks.

Pure numpy over `Track`/`Evidence` — no models, no video I/O — so every
check is unit-testable with fabricated tracks.

Conventions:
- distances are normalized by the frame diagonal;
- speeds are diagonals-per-second;
- image y grows downward, so "down" is +y and the ground line is a
  *large* y value.

Camera motion: generated videos often pan/shake, which would make every
static object "move". Per frame pair we estimate global motion as the
median displacement of all tracked centers and subtract it before any
velocity-based check.
"""

from __future__ import annotations

import numpy as np

from .config import Thresholds
from .tracks import (
    BONES,
    Evidence,
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    Track,
    Violation,
)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def camera_motion(evidence: Evidence) -> dict[int, np.ndarray]:
    """Median center displacement per frame transition (frame f -> f+1 keyed by f).

    With fewer than 3 tracks in a frame pair, the median cannot be told
    apart from genuine object motion, so no compensation is applied there.
    """
    disp: dict[int, list[np.ndarray]] = {}
    for tr in evidence.all_tracks:
        frames, centers = tr.frames_arr, tr.centers
        for i in range(len(frames) - 1):
            if frames[i + 1] - frames[i] == 1:
                disp.setdefault(int(frames[i]), []).append(centers[i + 1] - centers[i])
    return {f: np.median(np.stack(v), axis=0)
            for f, v in disp.items() if len(v) >= 3}


def _velocities(track: Track, diag: float, fps: float,
                cam: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Camera-compensated center velocity (diag/s) per consecutive frame pair.

    Returns (start_frames, velocities[N,2]); non-consecutive gaps are skipped.
    """
    frames, centers = track.frames_arr, track.centers
    out_f, out_v = [], []
    for i in range(len(frames) - 1):
        if frames[i + 1] - frames[i] != 1:
            continue
        v = centers[i + 1] - centers[i] - cam.get(int(frames[i]), np.zeros(2))
        out_f.append(int(frames[i]))
        out_v.append(v / diag * fps)
    if not out_f:
        return np.empty(0, dtype=np.int64), np.empty((0, 2))
    return np.asarray(out_f, dtype=np.int64), np.stack(out_v)


def estimate_ground_y(evidence: Evidence) -> float | None:
    """Ground line in pixels: high percentile of confident ankle positions."""
    ys = []
    for tr in evidence.person_tracks:
        k = tr.kpts_arr
        if k is None:
            continue
        for idx in (L_ANKLE, R_ANKLE):
            pts = k[:, idx, :]
            ys.extend(pts[pts[:, 2] > 0.5, 1].tolist())
    if len(ys) < 5:
        return None
    return float(np.percentile(ys, 90))


def _runs(mask: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Indices of consecutive True runs of at least min_len."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                runs.append(np.arange(start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append(np.arange(start, len(mask)))
    return runs


def _severity(ratio: float) -> float:
    """Map threshold-exceedance ratio (>=1) to severity in [0.5, 1]."""
    return float(np.clip(0.5 + 0.25 * (ratio - 1.0), 0.5, 1.0))


# --------------------------------------------------------------------------
# 1. trajectory jumps (teleportation)
# --------------------------------------------------------------------------

def check_trajectory_jumps(evidence: Evidence, th: Thresholds,
                           cam: dict[int, np.ndarray]) -> list[Violation]:
    violations = []
    for tr in evidence.all_tracks:
        frames, vel = _velocities(tr, evidence.diag, evidence.fps, cam)
        if len(frames) == 0:
            continue
        speed = np.linalg.norm(vel, axis=1)
        disp = speed / evidence.fps  # back to per-frame displacement (diag)
        bad = (speed > th.jump_speed) & (disp > th.jump_min_disp)
        for i in np.flatnonzero(bad):
            kind = "person" if tr.is_person else tr.label
            violations.append(Violation(
                type="trajectory_jump",
                severity=_severity(speed[i] / th.jump_speed),
                frames=[int(frames[i]), int(frames[i]) + 1],
                reason=(f"The {kind} center jumps {disp[i]:.0%} of the frame "
                        f"diagonal between consecutive frames without a plausible "
                        f"motion path."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 2. bone length consistency (body deformation)
# --------------------------------------------------------------------------

def check_bone_lengths(evidence: Evidence, th: Thresholds) -> list[Violation]:
    """Foreshortening can only *shorten* a 2D bone projection, so the robust
    per-track maximum (a high percentile, scale-normalized by bbox height)
    approximates the true length. A bone exceeding it by a wide margin is
    genuinely stretching."""
    violations = []
    for tr in evidence.person_tracks:
        k = tr.kpts_arr
        if k is None or len(k) < th.bone_min_samples:
            continue
        frames = tr.frames_arr
        # Scale by torso length (mid-shoulder to mid-hip): unlike bbox height
        # it is not distorted when the person is partially out of frame.
        mid_sh = (k[:, L_SHOULDER, :2] + k[:, R_SHOULDER, :2]) / 2
        mid_hip = (k[:, L_HIP, :2] + k[:, R_HIP, :2]) / 2
        torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
        torso_ok = ((k[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], 2]
                     > th.kpt_conf).all(axis=1)
                    & (torso > th.bone_min_len * evidence.diag))
        person_scale = np.maximum(torso, 1e-3)
        for a, b, name in BONES:
            conf_ok = (k[:, a, 2] > th.kpt_conf) & (k[:, b, 2] > th.kpt_conf)
            length = np.linalg.norm(k[:, a, :2] - k[:, b, :2], axis=1)
            valid = (conf_ok & torso_ok
                     & (length > th.bone_min_len * evidence.diag))
            if valid.sum() < th.bone_min_samples:
                continue
            norm_len = length / person_scale
            ref = float(np.percentile(norm_len[valid], th.bone_ref_pct))
            if ref < 1e-3:
                continue
            excess = norm_len / ref - 1.0
            bad = valid & (excess > th.bone_stretch_tol)
            if bad.sum() >= th.bone_min_events:
                worst = float(excess[bad].max())
                violations.append(Violation(
                    type="body_deformation",
                    severity=_severity((worst / th.bone_stretch_tol + 1.0) / 2),
                    frames=[int(f) for f in frames[bad]],
                    reason=(f"The {name} of person #{tr.track_id} stretches to "
                            f"{1 + worst:.0%} of its own established length, "
                            f"which is anatomically impossible."),
                    track_id=tr.track_id, label=tr.label,
                ))
    return violations


# --------------------------------------------------------------------------
# 3. foot skating
# --------------------------------------------------------------------------

def check_foot_skate(evidence: Evidence, th: Thresholds,
                     cam: dict[int, np.ndarray],
                     ground_y: float | None) -> list[Violation]:
    if ground_y is None:
        return []
    violations = []
    diag, fps = evidence.diag, evidence.fps
    for tr in evidence.person_tracks:
        k = tr.kpts_arr
        if k is None or len(k) < th.skate_min_frames + 1:
            continue
        frames = tr.frames_arr
        for idx, side in ((L_ANKLE, "left"), (R_ANKLE, "right")):
            pts = k[:, idx, :]
            ok_f, vx, vy, grounded = [], [], [], []
            for i in range(len(frames) - 1):
                if frames[i + 1] - frames[i] != 1:
                    continue
                if pts[i, 2] < th.kpt_conf or pts[i + 1, 2] < th.kpt_conf:
                    continue
                v = (pts[i + 1, :2] - pts[i, :2]
                     - cam.get(int(frames[i]), np.zeros(2))) / diag * fps
                ok_f.append(int(frames[i]))
                vx.append(abs(v[0]))
                vy.append(abs(v[1]))
                grounded.append(abs(pts[i, 1] - ground_y) < th.ground_band * diag)
            if not ok_f:
                continue
            vx, vy = np.asarray(vx), np.asarray(vy)
            mask = np.asarray(grounded) & (vy < th.skate_max_vy) & (vx > th.skate_min_vx)
            for run in _runs(mask, th.skate_min_frames):
                worst = float(vx[run].max())
                violations.append(Violation(
                    type="foot_skate",
                    severity=_severity(worst / th.skate_min_vx) * 0.9,
                    frames=[ok_f[i] for i in run],
                    reason=(f"The {side} foot of person #{tr.track_id} appears "
                            f"grounded but slides horizontally at "
                            f"{worst:.2f} diag/s, indicating unstable contact "
                            f"physics."),
                    track_id=tr.track_id, label=tr.label,
                ))
    return violations


# --------------------------------------------------------------------------
# 4. contact coherence (object moves with nobody touching it)
# --------------------------------------------------------------------------

def _min_wrist_dist(persons: list[Track], frame: int,
                    box: np.ndarray, kpt_conf: float) -> float:
    """Min distance (px) from any confident wrist to the object's bbox."""
    best = np.inf
    x1, y1, x2, y2 = box
    for tr in persons:
        k = tr.kpts_arr
        if k is None:
            continue
        pos = np.searchsorted(tr.frames_arr, frame)
        if pos >= len(tr.frames_arr) or tr.frames_arr[pos] != frame:
            continue
        for idx in (L_WRIST, R_WRIST):
            x, y, c = k[pos, idx]
            if c < kpt_conf:
                continue
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            best = min(best, float(np.hypot(dx, dy)))
    return best


def check_contact_coherence(evidence: Evidence, th: Thresholds,
                            cam: dict[int, np.ndarray]) -> list[Violation]:
    if not evidence.person_tracks:
        return []
    violations = []
    diag = evidence.diag
    for tr in evidence.object_tracks:
        frames, vel = _velocities(tr, diag, evidence.fps, cam)
        if len(frames) == 0:
            continue
        speed = np.linalg.norm(vel, axis=1)
        # Free fall is a legitimate reason for contact-free motion.
        falling = (vel[:, 1] > 0) & (vel[:, 1] / np.maximum(speed, 1e-6) > th.fall_vy_ratio)
        moving = (speed > th.carried_min_speed) & ~falling
        if not moving.any():
            continue
        boxes = tr.boxes_arr
        no_contact = np.zeros(len(frames), dtype=bool)
        for i in np.flatnonzero(moving):
            d = _min_wrist_dist(evidence.person_tracks, int(frames[i]),
                                boxes[i], th.kpt_conf)
            no_contact[i] = d / diag > th.contact_dist
        for run in _runs(moving & no_contact, th.contact_min_frames):
            worst = float(speed[run].max())
            violations.append(Violation(
                type="contact_incoherence",
                severity=_severity(worst / th.carried_min_speed) * 0.9,
                frames=[int(frames[i]) for i in run],
                reason=(f"The {tr.label} moves at {worst:.2f} diag/s but no hand "
                        f"is close enough to explain the motion, and it is not "
                        f"in free fall."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 5. object persistence (disappear / reappear elsewhere)
# --------------------------------------------------------------------------

def check_object_persistence(evidence: Evidence, th: Thresholds) -> list[Violation]:
    violations = []
    diag = evidence.diag
    max_gap = max(3, int(th.persist_gap_s * evidence.fps))
    for tr in evidence.all_tracks:
        frames, centers = tr.frames_arr, tr.centers
        for i in range(len(frames) - 1):
            gap = int(frames[i + 1] - frames[i]) - 1
            if gap < max_gap:
                continue
            disp = float(np.linalg.norm(centers[i + 1] - centers[i]) / diag)
            if disp > th.persist_reappear_disp:
                kind = "person" if tr.is_person else tr.label
                violations.append(Violation(
                    type="object_disappearance",
                    severity=_severity(disp / th.persist_reappear_disp) * 0.85,
                    frames=[int(frames[i]), int(frames[i + 1])],
                    reason=(f"The {kind} disappears for {gap} frames and "
                            f"reappears {disp:.0%} of the frame diagonal away."),
                    track_id=tr.track_id, label=tr.label,
                ))
    return violations


# --------------------------------------------------------------------------
# 6. gravity suspicion (unsupported hovering object)
# --------------------------------------------------------------------------

def check_gravity(evidence: Evidence, th: Thresholds,
                  cam: dict[int, np.ndarray],
                  ground_y: float | None) -> list[Violation]:
    if ground_y is None:
        return []
    violations = []
    diag, fps = evidence.diag, evidence.fps
    min_frames = max(2, int(th.float_min_s * fps))
    for tr in evidence.object_tracks:
        frames, vel = _velocities(tr, diag, evidence.fps, cam)
        if len(frames) < min_frames:
            continue
        speed = np.linalg.norm(vel, axis=1)
        boxes = tr.boxes_arr
        hovering = np.zeros(len(frames), dtype=bool)
        for i, f in enumerate(frames):
            bottom_y = boxes[i, 3]  # bbox bottom edge
            if ground_y - bottom_y < th.float_ground_clearance * diag:
                continue  # near the ground (or below the line) — fine
            if speed[i] > th.float_max_speed:
                continue
            center = (boxes[i, :2] + boxes[i, 2:]) / 2
            near_person = False
            for p in evidence.person_tracks:
                pos = np.searchsorted(p.frames_arr, int(f))
                if pos >= len(p.frames_arr) or p.frames_arr[pos] != f:
                    continue
                if np.linalg.norm(p.centers[pos] - center) / diag < th.float_person_dist:
                    near_person = True
                    break
            hovering[i] = not near_person
        for run in _runs(hovering, min_frames):
            violations.append(Violation(
                type="gravity_suspicion",
                severity=0.6,
                frames=[int(frames[i]) for i in run],
                reason=(f"The {tr.label} hovers above the ground for "
                        f"{len(run) / fps:.1f}s with no support and no person "
                        f"nearby, which violates gravity."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 7. materialization / vanishing (objects appearing out of nowhere)
# --------------------------------------------------------------------------

def _near_edge(box, width: int, height: int, margin: float) -> bool:
    x1, y1, x2, y2 = box
    return (x1 < margin * width or y1 < margin * height
            or x2 > (1 - margin) * width or y2 > (1 - margin) * height)


def _near_any_person(evidence: Evidence, frame: int, center: np.ndarray,
                     max_dist: float, exclude_id: int | None = None) -> bool:
    for p in evidence.person_tracks:
        if exclude_id is not None and p.track_id == exclude_id:
            continue
        pos = np.searchsorted(p.frames_arr, frame)
        lo, hi = max(0, pos - 1), min(len(p.frames_arr), pos + 1)
        for i in range(lo, hi):
            if abs(int(p.frames_arr[i]) - frame) <= 2:
                if np.linalg.norm(p.centers[i] - center) / evidence.diag < max_dist:
                    return True
    return False


def check_materialization(evidence: Evidence, th: Thresholds) -> list[Violation]:
    violations = []
    diag = evidence.diag
    margin_f = int(th.mater_margin_s * evidence.fps)
    gap_f = max(2, int(th.idswitch_gap_s * evidence.fps))
    tracks = evidence.all_tracks
    starts = [(t.frames_arr[0], t.centers[0], t.track_id) for t in tracks if len(t)]
    ends = [(t.frames_arr[-1], t.centers[-1], t.track_id) for t in tracks if len(t)]

    def id_switch(frame: int, center: np.ndarray, others, tid: int) -> bool:
        """A track ending/starting nearby around the same time is the tracker
        losing the ID, not an object popping in or out of existence."""
        for f, c, other_id in others:
            if other_id == tid:
                continue
            if abs(int(f) - frame) <= gap_f and \
                    np.linalg.norm(c - center) / diag < th.idswitch_dist:
                return True
        return False

    for tr in tracks:
        kind = "person" if tr.is_person else tr.label
        first_f, first_c = int(tr.frames_arr[0]), tr.centers[0]
        last_f, last_c = int(tr.frames_arr[-1]), tr.centers[-1]
        exclude = tr.track_id if tr.is_person else None

        if (first_f > margin_f
                and not _near_edge(tr.boxes[0], evidence.width, evidence.height,
                                   th.edge_margin)
                and not _near_any_person(evidence, first_f, first_c,
                                         th.mater_person_dist, exclude)
                and not id_switch(first_f, first_c, ends, tr.track_id)):
            violations.append(Violation(
                type="object_materialization",
                severity=0.75,
                frames=[first_f],
                reason=(f"A {kind} materializes out of nowhere at frame "
                        f"{first_f}, away from frame edges and from anyone who "
                        f"could have introduced it."),
                track_id=tr.track_id, label=tr.label,
            ))

        if (last_f < evidence.n_frames - margin_f
                and not _near_edge(tr.boxes[-1], evidence.width, evidence.height,
                                   th.edge_margin)
                and not _near_any_person(evidence, last_f, last_c,
                                         th.mater_person_dist, exclude)
                and not id_switch(last_f, last_c, starts, tr.track_id)):
            violations.append(Violation(
                type="object_materialization",
                severity=0.70,
                frames=[last_f],
                reason=(f"The {kind} vanishes into thin air at frame {last_f}, "
                        f"away from frame edges and with nobody nearby to "
                        f"remove it."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 8. person levitation
# --------------------------------------------------------------------------

def check_levitation(evidence: Evidence, th: Thresholds,
                     cam: dict[int, np.ndarray],
                     ground_y: float | None) -> list[Violation]:
    if ground_y is None:
        return []
    violations = []
    diag, fps = evidence.diag, evidence.fps
    min_frames = max(3, int(th.levit_min_s * fps))
    for tr in evidence.person_tracks:
        k = tr.kpts_arr
        if k is None or len(k) < min_frames + 1:
            continue
        frames = tr.frames_arr
        ok_f, airborne, slow_vy = [], [], []
        for i in range(len(frames) - 1):
            if frames[i + 1] - frames[i] != 1:
                continue
            ank = k[i, [L_ANKLE, R_ANKLE]]
            if (ank[:, 2] < th.kpt_conf).any():
                continue
            above = (ground_y - ank[:, 1] > th.levit_clearance * diag).all()
            v = (tr.centers[i + 1] - tr.centers[i]
                 - cam.get(int(frames[i]), np.zeros(2))) / diag * fps
            ok_f.append(int(frames[i]))
            airborne.append(above)
            slow_vy.append(abs(v[1]) < th.levit_max_vy)
        if not ok_f:
            continue
        mask = np.asarray(airborne) & np.asarray(slow_vy)
        for run in _runs(mask, min_frames):
            violations.append(Violation(
                type="levitation",
                severity=0.85,
                frames=[ok_f[i] for i in run],
                reason=(f"Person #{tr.track_id} hangs in the air for "
                        f"{len(run) / fps:.1f}s with both feet off the ground "
                        f"and no vertical motion — too long and too static for "
                        f"a jump."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 9. telekinesis (object motion mirrors a distant hand gesture)
# --------------------------------------------------------------------------

def _nearest_wrist(persons: list[Track], frame: int, point: np.ndarray,
                   kpt_conf: float):
    """(person_track, wrist_index, position, distance) of the closest wrist."""
    best = None
    for tr in persons:
        k = tr.kpts_arr
        if k is None:
            continue
        pos = np.searchsorted(tr.frames_arr, frame)
        if pos >= len(tr.frames_arr) or tr.frames_arr[pos] != frame:
            continue
        for idx in (L_WRIST, R_WRIST):
            x, y, c = k[pos, idx]
            if c < kpt_conf:
                continue
            d = float(np.linalg.norm(np.array([x, y]) - point))
            if best is None or d < best[3]:
                best = (tr, idx, np.array([x, y]), d)
    return best


def check_telekinesis(evidence: Evidence, th: Thresholds,
                      cam: dict[int, np.ndarray]) -> list[Violation]:
    if not evidence.person_tracks:
        return []
    violations = []
    diag, fps = evidence.diag, evidence.fps
    for tr in evidence.object_tracks:
        frames, vel = _velocities(tr, diag, fps, cam)
        if len(frames) < th.tk_min_frames:
            continue
        speed = np.linalg.norm(vel, axis=1)
        centers = tr.centers
        f2i = {int(f): i for i, f in enumerate(tr.frames_arr)}
        mask = np.zeros(len(frames), dtype=bool)
        for i, f in enumerate(frames):
            if speed[i] < th.tk_min_speed:
                continue
            hit = _nearest_wrist(evidence.person_tracks, int(f),
                                 centers[f2i[int(f)]], th.kpt_conf)
            if hit is None or hit[3] / diag < th.tk_dist:
                continue  # no confident wrist, or close enough for real contact
            person, idx, w_now, _ = hit
            pos = np.searchsorted(person.frames_arr, int(f) + 1)
            if pos >= len(person.frames_arr) or person.frames_arr[pos] != f + 1:
                continue
            w_next = person.kpts_arr[pos, idx]
            if w_next[2] < th.kpt_conf:
                continue
            wv = (w_next[:2] - w_now - cam.get(int(f), np.zeros(2))) / diag * fps
            wn, on = np.linalg.norm(wv), np.linalg.norm(vel[i])
            if wn < 0.1 or on < 1e-6:
                continue  # hand barely moving -> not a gesture
            if float(np.dot(wv, vel[i]) / (wn * on)) > th.tk_corr:
                mask[i] = True
        for run in _runs(mask, th.tk_min_frames):
            violations.append(Violation(
                type="telekinesis_suspicion",
                severity=0.8,
                frames=[int(frames[i]) for i in run],
                reason=(f"The {tr.label} moves in lockstep with a hand gesture "
                        f"for {len(run) / fps:.1f}s while the hand is too far "
                        f"away to touch it — motion without physical contact."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# 10. object deformation (rigid object morphing)
# --------------------------------------------------------------------------

def check_object_deformation(evidence: Evidence, th: Thresholds) -> list[Violation]:
    violations = []
    for tr in evidence.object_tracks:
        boxes = tr.boxes_arr
        if len(boxes) < th.deform_min_events + 1:
            continue
        frames = tr.frames_arr
        w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-3)
        h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-3)
        log_aspect = np.log(w / h)
        bad_frames, jumps = [], []
        for i in range(len(frames) - 1):
            if frames[i + 1] - frames[i] != 1:
                continue
            d = abs(float(log_aspect[i + 1] - log_aspect[i]))
            if d > th.deform_log_aspect:
                bad_frames.append(int(frames[i + 1]))
                jumps.append(d)
        if len(bad_frames) >= th.deform_min_events:
            worst = max(jumps)
            violations.append(Violation(
                type="object_deformation",
                severity=_severity(worst / th.deform_log_aspect) * 0.85,
                frames=bad_frames,
                reason=(f"The {tr.label} repeatedly snaps its shape (aspect "
                        f"ratio changes up to {np.exp(worst) - 1:.0%} per "
                        f"frame), which a rigid object cannot do."),
                track_id=tr.track_id, label=tr.label,
            ))
    return violations


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run_all_checks(evidence: Evidence,
                   th: Thresholds | None = None) -> list[Violation]:
    from .config import DEFAULT_THRESHOLDS
    th = th or DEFAULT_THRESHOLDS
    cam = camera_motion(evidence)
    ground_y = estimate_ground_y(evidence)
    violations: list[Violation] = []
    violations += check_trajectory_jumps(evidence, th, cam)
    violations += check_bone_lengths(evidence, th)
    violations += check_foot_skate(evidence, th, cam, ground_y)
    violations += check_contact_coherence(evidence, th, cam)
    violations += check_object_persistence(evidence, th)
    violations += check_gravity(evidence, th, cam, ground_y)
    violations += check_materialization(evidence, th)
    violations += check_levitation(evidence, th, cam, ground_y)
    violations += check_telekinesis(evidence, th, cam)
    violations += check_object_deformation(evidence, th)
    violations.sort(key=lambda v: v.severity, reverse=True)
    return violations
