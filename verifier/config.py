"""Tunable thresholds and scoring weights.

All spatial quantities are normalized by the frame diagonal, all speeds are
in diagonals-per-second, so thresholds are resolution- and fps-independent.
"""

from dataclasses import dataclass, field


@dataclass
class Thresholds:
    # --- extraction ---
    det_conf: float = 0.30          # min detection confidence
    kpt_conf: float = 0.50          # min keypoint confidence to use it
    max_frames: int = 900           # hard cap on processed frames

    # --- trajectory jump (teleportation) ---
    jump_speed: float = 3.5         # diag/s between consecutive frames
    jump_min_disp: float = 0.10     # absolute displacement floor (diag)

    # --- bone length consistency ---
    # 2D projected bone length legitimately shrinks with foreshortening
    # (arm swinging toward the camera), so frame-to-frame change is noise.
    # What cannot happen is a bone exceeding its own established maximum:
    # we flag lengths above a robust per-track reference by bone_stretch_tol.
    bone_stretch_tol: float = 0.40  # allowed excess over the reference length
    bone_ref_pct: float = 90.0      # percentile used as reference length
    bone_min_events: int = 2        # how many bad frames before flagging
    bone_min_samples: int = 10      # frames needed to establish the reference
    bone_min_len: float = 0.02      # skip segments shorter than this (diag):
                                    # keypoint jitter on tiny/distant people
                                    # makes ratios meaningless

    # --- foot skating ---
    ground_band: float = 0.035      # ankle within this of ground line (diag)
    skate_max_vy: float = 0.30      # "grounded" vertical speed cap (diag/s)
    skate_min_vx: float = 0.60      # horizontal slide speed to flag (diag/s)
    skate_min_frames: int = 3       # consecutive frames required

    # --- contact coherence ---
    contact_dist: float = 0.06      # wrist-to-object distance = "touching"
    carried_min_speed: float = 0.50 # object speed that demands an explanation
    contact_min_frames: int = 4     # consecutive incoherent frames required
    fall_vy_ratio: float = 0.6      # |vy_down| / speed above this = free fall

    # --- object persistence ---
    persist_gap_s: float = 0.20     # tracking gap longer than this (seconds)
    persist_reappear_disp: float = 0.12  # and reappearing farther than this

    # --- gravity / floating ---
    float_ground_clearance: float = 0.10  # object clearly above ground line
    float_person_dist: float = 0.15       # no person nearby
    float_max_speed: float = 0.08         # hovering speed cap (diag/s)
    float_min_s: float = 1.0              # hover duration to flag (seconds)

    # --- materialization / vanishing (objects appearing out of nowhere) ---
    mater_margin_s: float = 0.5     # ignore spawns/vanishes this close to video start/end
    edge_margin: float = 0.04       # bbox this close to a frame border = entering/leaving
    mater_person_dist: float = 0.12 # near a person -> could be put down / revealed
    idswitch_dist: float = 0.10     # vanish+spawn this close = tracker ID switch, not magic
    idswitch_gap_s: float = 0.3     # ...within this time window

    # --- person levitation ---
    levit_clearance: float = 0.05   # ankles above the ground line by this (diag)
    levit_min_s: float = 0.8        # sustained airtime (real jumps are shorter)
    levit_max_vy: float = 0.25      # near-zero vertical speed while airborne (diag/s)

    # --- telekinesis (object follows a distant hand gesture) ---
    tk_min_speed: float = 0.35      # object speed worth explaining (diag/s)
    tk_dist: float = 0.12           # wrist farther than this = no contact (diag)
    tk_corr: float = 0.85           # cosine similarity of object vs wrist velocity
    tk_min_frames: int = 5          # consecutive correlated frames required

    # --- object deformation (rigid object morphing) ---
    deform_log_aspect: float = 0.22 # per-frame |Δ log(w/h)| of the bbox
    deform_min_events: int = 3      # repeated shape snaps before flagging

    # --- gate 2 (semantic reviewer) ---
    gate2_model: str = "gemini-3.5-flash"
    gate2_frames: int = 10          # uniformly sampled frames sent to the VLM
    gate2_max_side: int = 1024      # frames are downscaled to this long edge
    gate2_min_report_severity: float = 0.75  # ignore weaker VLM notes
    gate2_severity_scale: float = 0.40       # make Gate 2 advisory
    gate2_severity_cap: float = 0.40         # never dominate Gate 1
                                      # finding lowers the score but can never
                                      # hard-reject on its own — the
                                      # deterministic gate stays the judge

    # --- decision ---
    accept_score: float = 0.72
    critical_severity: float = 0.85


# Weight of the worst violation of each type in the final score.
WEIGHTS: dict[str, float] = {
    # gate 1 — formal (deterministic) checks
    "trajectory_jump": 0.25,
    "contact_incoherence": 0.20,
    "body_deformation": 0.20,
    "object_materialization": 0.20,
    "telekinesis_suspicion": 0.15,
    "levitation": 0.15,
    "object_deformation": 0.15,
    "foot_skate": 0.15,
    "object_disappearance": 0.10,
    "gravity_suspicion": 0.10,
    # gate 2 — semantic (VLM) checks
    "anatomical_anomaly": 0.03,
    "magic_effect": 0.03,
    "object_morphing": 0.03,
    "impossible_gesture": 0.02,
    "scene_inconsistency": 0.02,
    "prompt_mismatch": 0.02,
}

DEFAULT_THRESHOLDS = Thresholds()
