"""Unit tests for the physics checks on fabricated tracks.

No models, no video files — we build Evidence by hand and assert that
each check fires on its target artifact and stays silent on clean motion.
"""

import numpy as np
import pytest

from verifier.checks import (
    camera_motion,
    check_bone_lengths,
    check_contact_coherence,
    check_foot_skate,
    check_gravity,
    check_levitation,
    check_materialization,
    check_object_deformation,
    check_object_persistence,
    check_telekinesis,
    check_trajectory_jumps,
    estimate_ground_y,
    run_all_checks,
)
from verifier.config import Thresholds
from verifier.gate2 import parse_gate2_response
from verifier.scoring import decide, plausibility_score
from verifier.tracks import Evidence, L_ANKLE, R_ANKLE, L_WRIST, Track

W, H, FPS = 1280, 720, 30.0
DIAG = float(np.hypot(W, H))
TH = Thresholds()


def make_evidence(persons=(), objects=(), n_frames=100):
    return Evidence(video_path="synthetic.mp4", fps=FPS, width=W, height=H,
                    n_frames=n_frames, person_tracks=list(persons),
                    object_tracks=list(objects))


def make_person(track_id=1, n=60, x0=300.0, dx=2.0, ground=650.0):
    """Walking person with a full, consistent skeleton."""
    tr = Track(track_id=track_id, label="person", is_person=True)
    for i in range(n):
        x = x0 + dx * i
        k = np.zeros((17, 3), dtype=np.float32)
        k[:, 2] = 0.9
        k[5] = (x - 40, 350, 0.9)   # shoulders
        k[6] = (x + 40, 350, 0.9)
        k[7] = (x - 55, 430, 0.9)   # elbows
        k[8] = (x + 55, 430, 0.9)
        k[9] = (x - 60, 505, 0.9)   # wrists
        k[10] = (x + 60, 505, 0.9)
        k[11] = (x - 25, 480, 0.9)  # hips
        k[12] = (x + 25, 480, 0.9)
        k[13] = (x - 25, 565, 0.9)  # knees
        k[14] = (x + 25, 565, 0.9)
        k[15] = (x - 25, ground, 0.9)  # ankles on the ground
        k[16] = (x + 25, ground, 0.9)
        tr.add(i, (x - 70, 320, x + 70, ground + 10), k)
    return tr


def make_object(track_id=10, label="box", n=60, x0=600.0, y0=500.0,
                dx=0.0, dy=0.0, size=60.0, f0=0):
    tr = Track(track_id=track_id, label=label, is_person=False)
    for i in range(n):
        x, y = x0 + dx * i, y0 + dy * i
        tr.add(f0 + i, (x, y, x + size, y + size))
    return tr


# ----------------------------------------------------------------------
# clean video -> no violations, accept
# ----------------------------------------------------------------------

def test_clean_scene_accepted():
    ev = make_evidence(persons=[make_person()],
                       objects=[make_object(dx=2.0, x0=360.0, y0=490.0)],
                       n_frames=60)
    # box rides next to the walking person's wrist -> coherent
    violations = run_all_checks(ev, TH)
    assert violations == []
    plausible, score, _ = decide(violations, TH)
    assert plausible and score == 1.0


# ----------------------------------------------------------------------
# individual checks
# ----------------------------------------------------------------------

def test_trajectory_jump_detected():
    tr = make_object(n=30)
    # teleport at frame 15: shift by 40% of the diagonal
    jump = 0.4 * DIAG
    tr.boxes = [(x1 + jump, y1, x2 + jump, y2) if i >= 15 else (x1, y1, x2, y2)
                for i, (x1, y1, x2, y2) in enumerate(tr.boxes)]
    ev = make_evidence(persons=[make_person(n=30)], objects=[tr])
    v = check_trajectory_jumps(ev, TH, camera_motion(ev))
    assert any(x.type == "trajectory_jump" and 14 in x.frames for x in v)


def test_camera_pan_not_flagged_as_jump():
    """Global pan moves everything; with >=3 tracks the median-displacement
    compensation must suppress it even when the pan alone would exceed the
    teleportation threshold."""
    pan = 0.12 * DIAG  # px/frame: above jump_min_disp without compensation
    p1 = make_person(track_id=1, n=40, dx=pan, x0=200.0)
    p2 = make_person(track_id=2, n=40, dx=pan, x0=500.0)
    o = make_object(n=40, dx=pan)
    ev = make_evidence(persons=[p1, p2], objects=[o])
    v = check_trajectory_jumps(ev, TH, camera_motion(ev))
    assert v == []


def test_bone_stretch_detected():
    tr = make_person(n=90)
    # stretch the left forearm x2 on frames 20-25 (short burst, so the
    # per-track reference length stays clean)
    for i in range(20, 26):
        k = tr.keypoints[i]
        k[L_WRIST, 0] -= 150  # wrist flies away from elbow
    ev = make_evidence(persons=[tr])
    v = check_bone_lengths(ev, TH)
    assert any(x.type == "body_deformation" for x in v)


def test_foot_skate_detected():
    tr = make_person(n=40, dx=0.0)
    # feet stay at ground height but slide fast horizontally, frames 10-20
    slide = TH.skate_min_vx * DIAG / FPS * 1.5  # px per frame, above threshold
    for i in range(10, 21):
        for idx in (L_ANKLE, R_ANKLE):
            tr.keypoints[i][idx, 0] = tr.keypoints[9][idx, 0] + slide * (i - 9)
    ev = make_evidence(persons=[tr])
    ground = estimate_ground_y(ev)
    v = check_foot_skate(ev, TH, camera_motion(ev), ground)
    assert any(x.type == "foot_skate" for x in v)


def test_contact_incoherence_detected():
    person = make_person(n=40, dx=0.0, x0=200.0)
    # box far from any wrist, moving fast horizontally (not falling)
    speed_px = TH.carried_min_speed * DIAG / FPS * 1.6
    box = make_object(n=40, x0=900.0, y0=300.0, dx=speed_px)
    ev = make_evidence(persons=[person], objects=[box])
    v = check_contact_coherence(ev, TH, camera_motion(ev))
    assert any(x.type == "contact_incoherence" for x in v)


def test_free_fall_not_flagged():
    person = make_person(n=40, dx=0.0, x0=200.0)
    fall_px = TH.carried_min_speed * DIAG / FPS * 2.0
    box = make_object(n=40, x0=900.0, y0=100.0, dy=fall_px)  # straight down
    ev = make_evidence(persons=[person], objects=[box])
    v = check_contact_coherence(ev, TH, camera_motion(ev))
    assert v == []


def test_carried_box_not_flagged():
    person = make_person(n=40, dx=3.0)
    # box glued to the right wrist trajectory
    box = Track(track_id=11, label="box", is_person=False)
    for i in range(40):
        wx, wy = person.keypoints[i][10, 0], person.keypoints[i][10, 1]
        box.add(i, (wx, wy, wx + 60, wy + 60))
    ev = make_evidence(persons=[person], objects=[box])
    v = check_contact_coherence(ev, TH, camera_motion(ev))
    assert v == []


def test_object_disappearance_detected():
    tr = Track(track_id=12, label="box", is_person=False)
    for i in range(15):
        tr.add(i, (600, 500, 660, 560))
    # gone for 12 frames, reappears far away
    for i in range(27, 45):
        tr.add(i, (600 + 0.3 * DIAG, 500, 660 + 0.3 * DIAG, 560))
    ev = make_evidence(persons=[make_person(n=45)], objects=[tr])
    v = check_object_persistence(ev, TH)
    assert any(x.type == "object_disappearance" for x in v)


def test_floating_object_detected():
    person = make_person(n=60, dx=0.0, x0=200.0, ground=650.0)
    # box parked high above the ground line, far from the person
    box = make_object(n=60, x0=1000.0, y0=150.0)
    ev = make_evidence(persons=[person], objects=[box])
    ground = estimate_ground_y(ev)
    v = check_gravity(ev, TH, camera_motion(ev), ground)
    assert any(x.type == "gravity_suspicion" for x in v)


# ----------------------------------------------------------------------
# materialization / vanishing
# ----------------------------------------------------------------------

def test_materialization_detected():
    person = make_person(n=100, dx=0.0, x0=200.0)
    # box pops into existence at frame 30, mid-frame, far from the person
    box = make_object(n=60, f0=30, x0=600.0, y0=400.0)
    ev = make_evidence(persons=[person], objects=[box], n_frames=100)
    v = check_materialization(ev, TH)
    assert any(x.type == "object_materialization" and 30 in x.frames for x in v)


def test_edge_entry_not_flagged():
    person = make_person(n=100, dx=0.0, x0=200.0)
    # box enters from the left frame edge at frame 30 -- normal
    box = make_object(n=60, f0=30, x0=5.0, y0=400.0)
    ev = make_evidence(persons=[person], objects=[box], n_frames=100)
    v = check_materialization(ev, TH)
    assert not any(30 in x.frames for x in v)


def test_tracker_id_switch_not_flagged():
    # same box, tracker loses the ID at frame 30 and re-assigns a new one
    a = make_object(track_id=10, n=30, f0=0, x0=600.0, y0=400.0)
    b = make_object(track_id=11, n=68, f0=32, x0=605.0, y0=400.0)
    ev = make_evidence(objects=[a, b], n_frames=100)
    v = check_materialization(ev, TH)
    assert v == []


# ----------------------------------------------------------------------
# levitation
# ----------------------------------------------------------------------

def test_levitation_detected():
    stander = make_person(track_id=1, n=60, x0=200.0, ground=650.0)
    floater = make_person(track_id=2, n=60, x0=800.0, ground=500.0)  # in the air
    ev = make_evidence(persons=[stander, floater], n_frames=60)
    ground = estimate_ground_y(ev)
    v = check_levitation(ev, TH, camera_motion(ev), ground)
    assert any(x.type == "levitation" and x.track_id == 2 for x in v)
    assert not any(x.track_id == 1 for x in v)


def test_short_jump_not_flagged():
    stander = make_person(track_id=1, n=60, x0=200.0, ground=650.0)
    jumper = make_person(track_id=2, n=60, x0=800.0, ground=650.0)
    # airborne for only 10 frames (~0.33s) -- a plausible hop
    for i in range(25, 35):
        for idx in (L_ANKLE, R_ANKLE):
            jumper.keypoints[i][idx, 1] = 500.0
    ev = make_evidence(persons=[stander, jumper], n_frames=60)
    ground = estimate_ground_y(ev)
    v = check_levitation(ev, TH, camera_motion(ev), ground)
    assert v == []


# ----------------------------------------------------------------------
# telekinesis
# ----------------------------------------------------------------------

def test_telekinesis_detected():
    # person walks right; a distant box mirrors the hand motion exactly
    person = make_person(n=40, dx=25.0, x0=200.0)
    box = make_object(n=40, x0=600.0, y0=300.0, dx=25.0)
    ev = make_evidence(persons=[person], objects=[box], n_frames=40)
    v = check_telekinesis(ev, TH, camera_motion(ev))
    assert any(x.type == "telekinesis_suspicion" for x in v)


def test_static_hand_no_telekinesis():
    # object moves but the hands don't -- no gesture to correlate with
    person = make_person(n=40, dx=0.0, x0=200.0)
    box = make_object(n=40, x0=600.0, y0=300.0, dx=25.0)
    ev = make_evidence(persons=[person], objects=[box], n_frames=40)
    v = check_telekinesis(ev, TH, camera_motion(ev))
    assert v == []


# ----------------------------------------------------------------------
# object deformation
# ----------------------------------------------------------------------

def test_object_deformation_detected():
    tr = Track(track_id=13, label="box", is_person=False)
    for i in range(20):
        if i % 2 == 0:
            tr.add(i, (600, 400, 660, 460))    # square
        else:
            tr.add(i, (600, 400, 700, 440))    # wide and flat
    ev = make_evidence(objects=[tr], n_frames=20)
    v = check_object_deformation(ev, TH)
    assert any(x.type == "object_deformation" for x in v)


def test_rigid_object_not_flagged():
    ev = make_evidence(objects=[make_object(n=40, dx=3.0)], n_frames=40)
    assert check_object_deformation(ev, TH) == []


# ----------------------------------------------------------------------
# gate 2 response parsing
# ----------------------------------------------------------------------

def test_parse_gate2_response():
    data = {
        "violations": [
            {"type": "anatomical_anomaly", "severity": 0.9,
             "frame_numbers": [12, 13], "reason": "The person has three arms."},
            {"type": "magic_effect", "severity": 1.7,
             "frame_numbers": [40], "reason": "The box glows and floats."},
            {"type": "scene_inconsistency", "severity": 0.6,
             "frame_numbers": [50], "reason": "Weak uncertain note."},
            {"type": "not_a_real_type", "severity": 0.5,
             "frame_numbers": [1], "reason": "ignored"},
        ],
        "semantic_score": 0.3,
        "summary": "Multiple impossibilities.",
    }
    v = parse_gate2_response(data)
    assert len(v) == 2
    assert all(x.gate == "semantic" for x in v)
    # Gate 2 is advisory: weak notes are dropped and remaining severities
    # are scaled down so the VLM can never hard-reject on its own.
    assert all(x.severity <= 0.40 for x in v)
    assert v[0].severity == 0.4 and v[0].type == "magic_effect"
    assert sorted(f for x in v for f in x.frames) == [12, 13, 40]


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------

def test_scoring_and_decision():
    from verifier.tracks import Violation
    vs = [Violation(type="trajectory_jump", severity=0.9, frames=[1], reason="x"),
          Violation(type="foot_skate", severity=0.6, frames=[2], reason="y")]
    score = plausibility_score(vs)
    assert score == pytest.approx(1 - 0.25 * 0.9 - 0.15 * 0.6)
    plausible, _, reason = decide(vs, TH)
    assert not plausible                       # critical severity 0.9 > 0.85
    assert "Critical" in reason


def test_duplicate_types_use_worst_only():
    from verifier.tracks import Violation
    vs = [Violation(type="foot_skate", severity=0.5, frames=[1], reason="a"),
          Violation(type="foot_skate", severity=0.7, frames=[9], reason="b")]
    assert plausibility_score(vs) == pytest.approx(1 - 0.15 * 0.7)
