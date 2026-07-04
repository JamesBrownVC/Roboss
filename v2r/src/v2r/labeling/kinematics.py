"""Kinematic labeling: contact inference and subtask segmentation.

These are REAL algorithms operating on whatever artifacts are in the episode
workspace — synthetic-mode puppet data and real estimated data go through the
same code. Nothing here fabricates events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema.models import Captions, SceneTags, Segment, SegmentsFile, SourceTag
from ..schema.rotations import se3_from_quat_pos, se3_inverse, transform_points

FINGERTIPS = ["thumbTip", "indexFingerTip", "middleFingerTip",
              "ringFingerTip", "littleFingerTip", "wrist"]


# ---------------------------------------------------------------------------
# contact inference: signed distance fingertips <-> object mesh (spec 6.F)
# ---------------------------------------------------------------------------


def infer_contacts(
    hands: pd.DataFrame,
    tracks: pd.DataFrame,
    meshes: dict[str, "object"],   # object_id -> trimesh.Trimesh (in object frame)
    dist_on_m: float = 0.005,
    dist_off_m: float = 0.010,
    sustain_frames: int = 3,
) -> pd.DataFrame:
    """CONTACTS_COLUMNS rows for every (frame, hand, object).

    contact=true if min fingertip->mesh distance < dist_on_m sustained >=
    sustain_frames; hysteresis releases at dist_off_m. Penetration depth is
    recorded as a QA signal. All rows source='estimated' (geometric inference).
    """
    import trimesh

    rows = []
    for object_id, grp in tracks.groupby("object_id", sort=True):
        grp = grp.sort_values("frame").reset_index(drop=True)
        obj_frames = grp["frame"].to_numpy()
        T_obj = se3_from_quat_pos(
            grp[["qw", "qx", "qy", "qz"]].to_numpy(), grp[["px", "py", "pz"]].to_numpy()
        )
        T_inv = se3_inverse(T_obj)
        obj_conf = grp["conf"].to_numpy()
        mesh = meshes.get(str(object_id))
        pq = trimesh.proximity.ProximityQuery(mesh) if mesh is not None else None

        for hand in ("left", "right"):
            h = hands[(hands["hand"] == hand) & (hands["joint_name"].isin(FINGERTIPS))]
            if h.empty:
                continue
            piv_pos = {}
            piv_conf = {}
            for jn, jgrp in h.groupby("joint_name"):
                jgrp = jgrp.sort_values("frame")
                piv_pos[jn] = jgrp[["px", "py", "pz"]].to_numpy()
                piv_conf[jn] = jgrp["conf"].to_numpy()
            names = list(piv_pos)
            n = min(len(obj_frames), min(len(v) for v in piv_pos.values()))
            pts = np.stack([piv_pos[jn][:n] for jn in names], axis=1)      # (n, J, 3)
            tipconf = np.stack([piv_conf[jn][:n] for jn in names], axis=1)  # (n, J)

            # transform tips into the object frame, batch signed distance
            pts_obj = transform_points(T_inv[:n], pts)                     # (n, J, 3)
            flat = pts_obj.reshape(-1, 3)
            if pq is not None:
                signed = pq.signed_distance(flat)      # trimesh: positive INSIDE
                dist = (-signed).reshape(n, len(names))  # positive outside
            else:
                dist = (np.linalg.norm(flat, axis=1) - 0.05).reshape(n, len(names))
            min_dist = dist.min(axis=1)
            penetration = np.maximum(0.0, -min_dist)

            # hysteresis + sustain state machine
            contact = np.zeros(n, dtype=bool)
            below = min_dist < dist_on_m
            state = False
            run = 0
            for i in range(n):
                if not state:
                    run = run + 1 if below[i] else 0
                    if run >= sustain_frames:
                        state = True
                        contact[i - sustain_frames + 1: i + 1] = True
                else:
                    if min_dist[i] > dist_off_m:
                        state = False
                        run = 0
                contact[i] = state
            conf = np.clip(tipconf.mean(axis=1) * obj_conf[:n], 0.0, 1.0)
            t_arr = grp["t"].to_numpy()[:n]
            for i in range(n):
                rows.append({
                    "t": t_arr[i], "frame": int(obj_frames[i]), "hand": hand,
                    "object_id": str(object_id), "contact": bool(contact[i]),
                    "min_dist_m": float(max(min_dist[i], 0.0)),
                    "penetration_m": float(penetration[i]),
                    "conf": float(conf[i]), "valid": True,
                    "source": SourceTag.estimated.value,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["hand", "object_id", "frame"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# subtask segmentation: aperture + contact changepoints (spec 6.G)
# ---------------------------------------------------------------------------


def aperture_series(hands: pd.DataFrame, hand: str = "right") -> tuple[np.ndarray, np.ndarray]:
    h = hands[hands["hand"] == hand]
    tt = h[h["joint_name"] == "thumbTip"].sort_values("frame")
    it = h[h["joint_name"] == "indexFingerTip"].sort_values("frame")
    n = min(len(tt), len(it))
    if n == 0:
        return np.array([]), np.array([])
    ap = np.linalg.norm(
        tt[["px", "py", "pz"]].to_numpy()[:n] - it[["px", "py", "pz"]].to_numpy()[:n], axis=1
    )
    if n >= 5:  # light smoothing
        k = np.ones(5) / 5
        ap = np.convolve(ap, k, mode="same")
    return tt["t"].to_numpy()[:n], ap


def segment_episode(
    t: np.ndarray,
    contacts: pd.DataFrame,
    tracks: pd.DataFrame,
    hands: pd.DataFrame,
    verbs: list[str],
    hand: str = "right",
    min_len_s: float = 0.12,
) -> list[Segment]:
    """Build labeled segments from contact transitions + object/wrist kinematics."""
    n = len(t)
    if n == 0:
        return []

    # per-frame: is the hand in contact with anything; which object
    c = contacts[contacts["hand"] == hand] if not contacts.empty else contacts
    contact_any = np.zeros(n, dtype=bool)
    contact_obj = np.array([""] * n, dtype=object)
    if not c.empty:
        for oid, grp in c.groupby("object_id"):
            grp = grp.sort_values("frame")
            fr = grp["frame"].to_numpy()
            fl = grp["contact"].to_numpy(dtype=bool)
            m = fr < n
            newly = fl[m] & ~contact_any[fr[m]]
            contact_any[fr[m]] |= fl[m]
            contact_obj[fr[m][newly]] = str(oid)

    # object heights over time (for lift/lower discrimination)
    heights: dict[str, np.ndarray] = {}
    for oid, grp in tracks.groupby("object_id"):
        grp = grp.sort_values("frame")
        z = np.full(n, np.nan)
        fr = grp["frame"].to_numpy()
        m = fr < n
        z[fr[m]] = grp["pz"].to_numpy()[m]
        heights[str(oid)] = z

    # wrist -> nearest-object distance (for reach detection)
    wr = hands[(hands["hand"] == hand) & (hands["joint_name"] == "wrist")].sort_values("frame")
    wrist = np.full((n, 3), np.nan)
    fr = wr["frame"].to_numpy()
    m = fr < n
    wrist[fr[m]] = wr[["px", "py", "pz"]].to_numpy()[m]
    obj_dist = np.full(n, np.nan)
    for oid, grp in tracks.groupby("object_id"):
        grp = grp.sort_values("frame")
        p = np.full((n, 3), np.nan)
        fr = grp["frame"].to_numpy()
        m = fr < n
        p[fr[m]] = grp[["px", "py", "pz"]].to_numpy()[m]
        d = np.linalg.norm(wrist - p, axis=1)
        obj_dist = np.fmin(obj_dist, d)

    # contact rise/fall events
    rises = np.flatnonzero(contact_any[1:] & ~contact_any[:-1]) + 1
    falls = np.flatnonzero(~contact_any[1:] & contact_any[:-1]) + 1
    if contact_any[0]:
        rises = np.concatenate([[0], rises])
    if contact_any[-1]:
        falls = np.concatenate([falls, [n - 1]])

    def pick(v: str) -> str:
        return v if v in verbs else ("idle" if "idle" in verbs else verbs[0])

    segs: list[tuple[int, int, str, str]] = []  # (i0, i1, skill, text)
    cursor = 0
    for r, f in zip(rises, falls):
        oid = contact_obj[r] if contact_obj[r] else "object"
        # reach: window before contact where wrist-object distance decreases
        reach_start = cursor
        if r > cursor + 2:
            d = obj_dist[cursor:r]
            ok = np.isfinite(d)
            if ok.sum() >= 3 and (np.nanmax(d) - d[ok][-1]) > 0.05:
                drop = np.nanargmax(d)
                reach_start = cursor + int(drop)
            if reach_start > cursor + int(0.2 * 30):
                segs.append((cursor, reach_start, pick("idle"), "waiting"))
            segs.append((reach_start, r, pick("reach"), f"reach toward {oid}"))
        elif r > cursor:
            segs.append((cursor, r, pick("reach"), f"reach toward {oid}"))

        # contact window: grasp -> lift/hold (-> lower)
        z = heights.get(str(oid))
        window_label = pick("hold")
        text = f"hold {oid}"
        split = None
        if z is not None and np.isfinite(z[r:f + 1]).any():
            zw = z[r:f + 1]
            zmax_i = int(np.nanargmax(zw))
            rise = np.nanmax(zw) - zw[np.isfinite(zw)][0]
            if rise > 0.04:
                window_label = pick("lift")
                text = f"lift {oid}"
                if zmax_i < len(zw) - int(0.2 * 30):
                    split = r + zmax_i
        grasp_end = min(r + max(2, int(0.15 * 30)), f)
        segs.append((r, grasp_end, pick("grasp"), f"grasp {oid}"))
        if split is not None:
            segs.append((grasp_end, split, window_label, text))
            segs.append((split, f, pick("lower"), f"lower {oid}"))
        else:
            segs.append((grasp_end, f, window_label, text))
        rel_end = min(f + max(2, int(0.15 * 30)), n - 1)
        segs.append((f, rel_end, pick("release"), f"release {oid}"))
        cursor = rel_end
    if cursor < n - 1:
        segs.append((cursor, n - 1, pick("idle"), "waiting"))

    # merge tiny segments, convert to Segment models
    out: list[Segment] = []
    for i0, i1, skill, text in segs:
        if i1 <= i0:
            continue
        if t[i1] - t[i0] < min_len_s and out:
            prev = out[-1]
            out[-1] = Segment(start_s=prev.start_s, end_s=float(t[i1]),
                              skill=prev.skill, text=prev.text)
            continue
        out.append(Segment(start_s=float(t[i0]), end_s=float(t[i1]), skill=skill, text=text))
    if not out:
        out = [Segment(start_s=float(t[0]), end_s=float(t[-1]), skill=pick("idle"),
                       text="no contact events detected")]
    return out


def captions_from_segments(segments: list[Segment], object_ids: list[str],
                           source: SourceTag = SourceTag.estimated) -> Captions:
    skills = [s.skill for s in segments if s.skill != "idle"]
    uniq = list(dict.fromkeys(skills))
    objs = ", ".join(object_ids) if object_ids else "an object"
    if not uniq:
        short = "A mostly idle scene."
        medium = "A scene with no detected hand-object interaction."
        long = ("No contact events were detected between the tracked hands and objects; "
                "the episode is labeled idle throughout.")
    else:
        short = f"A person {uniq[0]}s {objs.split(',')[0]}."
        medium = f"A person performs: {', '.join(uniq)} involving {objs}."
        cyc = sum(1 for s in segments if s.skill == "grasp")
        long = (f"Tabletop manipulation with {len(segments)} subtasks over "
                f"{segments[-1].end_s - segments[0].start_s:.1f} s: "
                + "; ".join(f"{s.skill} [{s.start_s:.2f}-{s.end_s:.2f}s]" for s in segments)
                + (f". {cyc} grasp cycle(s) detected." if cyc else "."))
    return Captions(short=short, medium=medium, long=long, source=source)


def default_scene_tags(source: SourceTag = SourceTag.synthesized) -> SceneTags:
    return SceneTags(scene_type="tabletop", lighting="indoor", clutter=2,
                     surfaces=["table"], source=source)


def segments_file(segments: list[Segment], method: str,
                  source: SourceTag = SourceTag.estimated) -> SegmentsFile:
    return SegmentsFile(segments=segments, method=method, source=source)
