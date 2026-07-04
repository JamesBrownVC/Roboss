# MASTER PROMPT: Build "V2R Factory" (Video to Robot-Ready Data Pipeline)

You are a senior robotics infrastructure engineer. Your job is to build, from an empty repository, a production pipeline that converts raw human videos (monocular or multi-view) into fully labeled, physics-validated, training-ready robot learning datasets. Output format is LeRobot v3, with per-frame confidence, provenance, and an auditable yield report. Consumers of the output: teams training humanoid VLAs and whole-body controllers, quadruped (robot dog) teams, and manipulator/arm teams.

Domain context you should know: the target per-frame representation follows the EgoDex convention for hands (SE(3) for 25 joints per hand plus head and wrists), SMPL-X for the body, LeRobot v3 as the container, GMR as the humanoid retargeting reference, and ViPE as the geometry annotation engine. All are cited by repo below. Build exactly what is specified here; where a third-party README conflicts with this document on interfaces, this document wins; on installation details, the README wins.

---

## 1. Non-negotiable design principles

1. **Isolation over unification.** Every third-party research tool runs in its own environment (micromamba/pixi env, or Docker image when the build is fragile). Never attempt one shared environment: these repos have conflicting torch/CUDA/detectron dependencies by construction. Harmonization happens ONLY through the interchange schema on disk (Section 4).
2. **Contracts over code.** Every stage reads and writes the canonical episode workspace. Any stage is replaceable by anything that honors the contract. Stage wrappers translate contract to tool CLI and nothing else.
3. **Determinism and provenance.** Pin every repo to a commit hash and every model to a weights hash. Every stage writes a `manifest.json` (tool, commit, weights hash, config hash, runtime, status, metrics). A stage re-runs only if its input or config hash changed.
4. **Confidence is a first-class output.** Every estimated quantity ships with a per-frame confidence in [0,1], a validity mask, and a `source` tag. Never emit a number without its uncertainty channel.
5. **Fail loud, fail early, and count it.** Each stage has a quality gate. A rejected clip is a labeled funnel outcome (with failure reason), not an exception. The yield report is a product deliverable.
6. **Honesty about estimation.** This pipeline estimates from RGB. Nothing is called "ground truth" unless it comes from triangulated multi-view (Section 7) or an external capture source.

## 2. Repository skeleton

```
v2r/
  pyproject.toml            # orchestrator env only (uv)
  conventions.md            # Section 3, verbatim, enforced by schema lib
  config/
    pipeline.yaml           # stage toggles, thresholds, paths
    robots.yaml             # embodiment registry (Section 6.H)
    verbs.yaml              # fixed caption verb vocabulary
    qa.yaml                 # gate thresholds
    licensing.yaml          # permissive_only flag, license audit map
  src/v2r/
    schema/                 # pydantic v2 models + parquet/npz IO for every artifact
    stages/                 # one thin wrapper per stage (subprocess into stage env)
    orchestrator/           # CLI, DAG runner, manifests, locks
    qa/                     # cross-checks, yield report generator
    export/                 # LeRobot v3 writer, EgoDex-mirror writer
  envs/                     # per-stage env specs (pixi.toml / Dockerfile each)
  assets/
    body_models/            # SMPL-X, MANO (operator-provided, never scraped)
    robots/                 # URDF/MJCF + meshes per embodiment
  tests/
    data/sample.mp4         # 30 s test clip
  workspaces/               # episode workspaces (gitignored)
```

## 3. Global conventions (write to conventions.md; enforce in src/v2r/schema)

- Units: SI everywhere. Meters, seconds, radians, kilograms.
- World frame: right-handed, Z-up, gravity (0, 0, -9.81). Origin: camera position at the first tracked keyframe; X axis: horizontal projection of the initial camera forward vector. Deterministic per episode.
- Camera frame: OpenCV convention (X right, Y down, Z forward). Store extrinsics as `T_world_cam` (camera-to-world, 4x4, row-major).
- Rotations: quaternions scalar-first `wxyz` (MuJoCo convention) in all stored artifacts. SciPy is `xyzw`: conversions happen only in `schema/rotations.py` with unit tests. No exceptions.
- Canonical timeline: 30 Hz. Keep original timestamps; resampled values carry an `interpolated` flag.
- Human model: SMPL-X neutral, 10 betas stored (16 accepted), axis-angle pose. Hands additionally exported as SE(3) for 25 joints per hand in the EgoDex convention.
- Files: tabular data in Parquet; per-frame dense arrays in NPZ; depth as 16-bit PNG in millimeters with scale recorded in `camera.json`; meshes as GLB; video as MP4 (h264, yuv420p, 30 fps).
- Every kinematic Parquet table has, per logical quantity: value columns, `conf` (float 0-1), `valid` (bool), `source` (enum: `captured | estimated | triangulated | fused | synthesized`).
- IDs: `episode_id = {source_id}_{clip_idx:06d}`. All artifacts live under `workspaces/{episode_id}/`.

## 4. Interchange schema (episode workspace)

```
workspaces/{episode_id}/
  raw/            video.mp4, probe.json, consent.json
  geometry/       camera.json (intrinsics, model, depth_scale, scale_source)
                  poses.parquet (t, T_world_cam, conf, valid)
                  depth/{frame:06d}.png
                  scene.ply, scene_mesh.glb (fused static scene)
  human/          smplx.npz (betas, pose, transl, global_orient per frame + conf)
                  hands.parquet (25 joints x 2 hands, SE(3), conf, valid, source)
                  fusion_report.json (world alignment residuals)
  objects/        tracks.parquet (object_id, T_world_obj, conf, valid)
                  masks/{object_id}/...  meshes/{object_id}.glb
  contact/        contacts.parquet (hand, object_id, state, min_dist, conf)
  semantics/      segments.json (subtasks: start, end, skill, text)
                  captions.json (short/medium/long), scene_tags.json
  retargets/{robot}/  qpos.parquet, ee.parquet, mapping.json,
                      physics_report.json (per-frame violations, tracking_error)
  qa/             crosschecks.json, frames_review/, decision.json
  export/         lerobot/ (v3 repo fragment), egodex_mirror/{n}.hdf5+{n}.mp4
  manifests/      {stage}.manifest.json
```

## 5. Environment strategy and host requirements

- Host: Ubuntu 22.04+, NVIDIA driver + CUDA 12.x, one GPU with 24 GB VRAM recommended (12 GB workable with reduced batch/resolution), 2 TB NVMe, 64 GB RAM.
- Orchestrator env (uv, Python 3.11): pydantic v2, numpy, pyarrow, opencv-python-headless, scipy, trimesh, open3d, mujoco, lerobot, rerun-sdk, typer, rich, h5py, ffmpeg-python (ffmpeg installed at system level).
- Per-stage envs under `envs/{stage}/` via micromamba or pixi. Use Docker for stages with fragile native builds (FoundationPose, Isaac Lab). The orchestrator invokes stages as `micromamba run -n {env} python tool_entry.py ...` or `docker run ...`. Stage wrappers in `src/v2r/stages/` do contract translation only.
- Pin everything: git commit in env spec, weights sha256 in `manifests/`.

## 6. Stages: install and output contracts

For each stage: Purpose / Source / Install / Writes / Gate / Gotchas. Follow each repo's README for installation details at the pinned commit; ask the operator before downloading any weights that require registration or license acceptance.

### A. ingest
- Purpose: normalize input, protect PII, register consent.
- Tools: ffmpeg; PySceneDetect (pip) for shot segmentation; EgoBlur (github.com/facebookresearch/EgoBlur) or deface for face/plate blurring.
- Writes: `raw/video.mp4` (30 fps, yuv420p), `probe.json`, `consent.json` (consent_id, license, subject_ids). Blur is ON by default; export refuses to run without a consent record.
- Gate: resolution >= 720p, fps >= 15, duration 3 s to 15 min, no corrupt frames.

### B. geometry (the backbone)
- Purpose: per-frame camera intrinsics, `T_world_cam`, dense near-metric depth, static scene reconstruction.
- Source: ViPE (github.com/nv-tlabs/vipe). It estimates intrinsics, camera motion, and dense near-metric depth from unconstrained video, supports pinhole/fisheye/360, masks dynamic objects (GroundingDINO + SAM) so the solve locks onto the static scene, and runs at roughly 3-5 FPS on one GPU.
- Writes: `geometry/*` per Section 4. Fuse depth into `scene.ply` and a TSDF mesh (Open3D, in orchestrator env).
- Gate: tracked-frame ratio >= 0.9, median depth coverage >= 60 percent.
- Gotchas: ViPE scale is near-metric. Refine per-episode scale later against fused SMPL-X body height (Stage C) or a known-size object; record `scale_source` and the correction factor. Alternatives if needed: MegaSaM, VGGT, MASt3R-SLAM; COLMAP for static multi-view calibration.

### C. human_body (world-frame SMPL-X)
- Primary: GVHMR (github.com/zju3dv/GVHMR) for world-grounded body motion. Fallbacks: WHAM, TRAM.
- SMPL-X and MANO model files require registration at the MPI sites: STOP and ask the operator to place them in `assets/body_models/`. Never scrape them.
- Harmonization step (critical): GVHMR's world frame is not ViPE's. Solve a similarity transform (Umeyama) between GVHMR's camera trajectory and ViPE's, apply it to the body motion, write the residual into `human/fusion_report.json` and fold it into `conf`.
- Gate: mean 2D reprojection error of SMPL-X joints against a 2D detector (Sapiens or ViTPose, run in this stage's env) below `qa.yaml` threshold; jitter (joint acceleration) below threshold.

### D. hands
- Per-frame MANO: WiLoR (github.com/rolpotamias/WiLoR) or HaMeR (github.com/geopavlakos/hamer). Egocentric world-space option: HaWoR.
- Fuse MANO into SMPL-X at the wrists (replace SMPL-X hand articulation, solve the fixed wrist offset per episode). Export the 25-joint SE(3) tables per hand (EgoDex convention) into `human/hands.parquet`.
- Confidence: detector score x temporal-consistency score. Frames where the hand is occluded get `valid=false`, not interpolated values (interpolation allowed only with `interpolated` flag and capped gap length).

### E. objects
- Detection/tracking: Grounding DINO + SAM 2 (both permissive licenses), prompted with the manipulated-object list from Stage G (run G's detector pass first or iterate).
- 6DoF pose: FoundationPose (github.com/NVlabs/FoundationPose) with a mesh; mesh-free reconstruction: BundleSDF or SAM 3D, then FoundationPose tracking.
- LICENSE WARNING: FoundationPose and BundleSDF carry NVIDIA non-commercial licenses. Implement `licensing.permissive_only`: when true, replace with mask + depth ICP tracking (Open3D, custom) and mark lower confidence. Surface this in the license audit (Section 10).
- Writes: `objects/*` per Section 4, including collision-relevant untouched objects near the workspace (pose + coarse mesh).

### F. contact
- Geometric inference: signed distance between MANO hand mesh and object mesh per frame; `contact=true` if min distance < 5 mm sustained >= 3 frames; hysteresis on release. Penetration depth recorded as a QA signal.
- All contact fields are `source=estimated`. Do not fabricate forces; there are none in RGB.

### G. semantics
- Serve Qwen2.5-VL (7B minimum, larger if VRAM allows) via vLLM in its own env. Temperature 0, JSON-schema-constrained outputs.
- Subtask segmentation: changepoint detection on hand aperture + contact transitions (the human-video analog of gripper-transition segmentation), then one VLM call per segment for the skill label (must be from `config/verbs.yaml`) and step text.
- Episode captions: three lengths (short/medium/long). Scene tags: scene_type, lighting, clutter 1-5, surfaces. All auto; human QA only on samples.

### H. retarget (three adapters, selected by `class` in config/robots.yaml)
Registry entry: `{name, class, mjcf/urdf path, dof list, key_body_map, joint_limits, control_rate}`.
- H1 `humanoid_wholebody`: GMR (github.com/YanjieZe/GMR, MIT license). Real-time retargeting of SMPL-X/BVH/FBX/Xsens motion to 17+ humanoids (Unitree G1/H1, Booster T1, Fourier GR3, Talos, ...). GMR also documents a monocular path via GVHMR, which is exactly this pipeline's Stage C. Output `retargets/{robot}/qpos.parquet`. Convert to CSV where downstream trackers (BeyondMimic) expect it.
- H2 `ee_manipulator` (arms, arms on dogs, mobile manipulators): task-space retarget. Targets: wrist SE(3) trajectories + gripper aperture derived from hand aperture. Solver: mink (github.com/kevinzakka/mink, differential IK on MuJoCo) or cuRobo. Base motion, if the platform is mobile, from the human root trajectory as a velocity command channel.
- H3 `quadruped`: be honest about morphology. Two sources: (a) if the source video contains an actual animal: quadruped keypoints via SuperAnimal (DeepLabCut) or ViTPose-animal, 3D lift, then key-body IK to Go2/B2/Spot with a GMR-style key-body map in mink; (b) if the source is human video: DO NOT pretend human gait transfers. Extract base twist commands (root velocity, yaw rate) and, if an arm is mounted, the H2 EE channel. Mark gait as `source=synthesized` (the locomotion controller owns it).
- Every retarget row carries `retarget_method`, `retarget_version`, `provenance` (kinematic-retarget vs command-abstraction).

### I. physics_validate
- Tier 1 (all robots, ship first): MuJoCo replay of qpos in the orchestrator env. Checks: joint limits, self-collision, ground penetration, foot slide (contact-point drift per frame), velocity/acceleration limits, coarse balance heuristic for humanoids. Writes `physics_report.json` + per-frame violation mask. `physics_valid = no hard violations`.
- Tier 2 (humanoid/quadruped, later): track the motion with a whole-body tracking policy (BeyondMimic-style) in Isaac Lab or MuJoCo; record tracking error; motions a tracker follows within threshold get the premium flag. This mirrors the published methodology where retargeting quality is evaluated by training tracking policies with reward tuning suppressed.

### J. package (export)
- LeRobot v3 writer (`pip install lerobot`). Per-frame features: `observation.images.ego` (+ exo views), `observation.depth.ego`, `observation.camera_pose`, `human.head_pose`, `human.wrist_pose.{l,r}`, `human.hand_joints.{l,r}`, `human.smplx.body_pose`, `object.{id}.pose`, `contact.{l,r}`, `retarget.{robot}.qpos`, `action` aliased to the buyer's primary embodiment, standard indices, `subtask_index`.
- Episode metadata columns: instructions x3, subtasks list, smplx_betas, intrinsics, pose_quality, physics_valid + tracking_error, provenance fields, scene tags, consent_id, license.
- Sidecars: `egodex_mirror/` paired `{n}.hdf5 + {n}.mp4` (camera intrinsics + SE(3) pose tables, EgoDex layout), `assets/` object meshes, `qa/` report.
- Validation: round-trip load with `LeRobotDataset`, render a rerun snapshot per episode, assert schema.

### K. qa and yield
- Cross-checks: 3D joints reprojected into the image vs 2D detections; SMPL-X depth vs ViPE depth along the body silhouette; temporal jitter; quaternion norms; timestamp monotonicity. Thresholds in `config/qa.yaml`.
- Yield report (`qa/yield_report.md`, plus dataset-level aggregate): funnel counts ingested -> geometry_ok -> body_ok -> hands_ok -> objects_ok -> retarget_ok{robot} -> physics_ok{robot}, with a failure taxonomy. This report is a sales artifact; format it accordingly.

## 7. Multi-view mode (GT tier)

- Session layout: `raw/cams/{cam_id}/video.mp4`. Synchronization: hardware timecode if present; else audio cross-correlation (scipy); else a visual flash event. Store per-camera offset + sync confidence.
- Extrinsics: checkerboard (OpenCV) when the operator provides one; else joint SfM (COLMAP) over all views; intrinsics per camera from Stage B run per view.
- Triangulation upgrade: run 2D body (Sapiens/ViTPose) and 2D hands (WiLoR) per view, triangulate to 3D with RANSAC; these joints become `source=triangulated` with measured reprojection error as confidence. Fit SMPL-X to triangulated joints (standard IK fit). Ego view remains the image feature stream.
- Keep the monocular pipeline running in shadow mode on multi-view sessions and log its error against triangulation: this is the internal benchmark that continuously calibrates and improves the monocular product.

## 8. Orchestration

- CLI: `v2r run --episode <path|glob> --stages all --robots g1,go2,franka` (typer). DAG with per-stage idempotence (skip when manifest hash matches), retries with backoff, a simple GPU semaphore, JSONL logs.
- Parallelism: across episodes, serial within an episode. Start with a plain Python DAG + file locks; move to Snakemake/Prefect only if scale demands. Do not start with Kubernetes.

## 9. Build order (thin slice first; do not deviate)

- Phase 0: repo skeleton, schema library with IO + rotation tests, one fake stage, sample-video test harness. Accept: `v2r run` produces a valid empty workspace.
- Phase 1: ingest + ViPE + package. Accept: sample video becomes a loadable LeRobot repo containing video, camera pose, depth.
- Phase 2: body + hands + world-frame fusion + reprojection QA. Accept: rerun snapshot shows skeleton overlaid correctly; conf channels populated.
- Phase 3: objects + contact + semantics. Accept: segments.json and object tracks pass gates on the sample.
- Phase 4: retarget adapters + Tier-1 physics + yield report. Accept: G1 qpos replays in MuJoCo without hard violations on at least one sample; yield_report.md renders.
- Phase 5: multi-view + Tier-2 physics + EgoDex mirror. Accept: triangulated session beats monocular error on the same event; tracking error recorded.

## 10. Licensing and compliance guardrails (do not skip)

- SMPL-X and MANO model files are free for research; commercial use requires a commercial license (Meshcapade). The pipeline must run only with operator-provided model files and must record the license tier in every export.
- Generate `LICENSE_AUDIT.md` automatically: every repo and weights file with its license. Permissive allowlist (MIT/Apache/BSD): GMR, mink, SAM 2, Grounding DINO, LeRobot, MuJoCo, rerun. Non-commercial watchlist (verify before commercial use): FoundationPose, BundleSDF, some model weights (including certain VLM checkpoints). `licensing.permissive_only=true` must produce a fully permissive pipeline with degraded-confidence fallbacks.
- PII: face/plate blurring on by default; consent ledger required for export; source videos are never redistributed, only derived labels and, where the consent record allows, the blurred video.

## 11. What NOT to do

- No single mega-environment. No silent unit or quaternion-order conversions. No interpolation across long occlusions. No training custom models in v1 (integration first, learning later). No scraping registration-gated model files. No optimizing anything before the yield report exists. No calling estimated quantities ground truth.

## 12. Definition of done (v1)

- [ ] `v2r run --episode tests/data/sample.mp4 --robots g1` completes Phases 1-4 stages end to end.
- [ ] Output loads with `LeRobotDataset` and renders in rerun.
- [ ] Every kinematic table has conf/valid/source populated; manifests exist for every stage.
- [ ] Tier-1 physics report and yield report generated.
- [ ] LICENSE_AUDIT.md generated; export blocked without consent record.
- [ ] Unit tests: rotations, schema round-trip, timeline resampling, Umeyama alignment.
