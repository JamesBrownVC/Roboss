# V2R Global Conventions

These conventions are normative for every artifact in every episode workspace.
They are enforced programmatically by `src/v2r/schema`.

- Units: SI everywhere. Meters, seconds, radians, kilograms.
- World frame: right-handed, Z-up, gravity (0, 0, -9.81). Origin: camera position
  at the first tracked keyframe; X axis: horizontal projection of the initial
  camera forward vector. Deterministic per episode.
- Camera frame: OpenCV convention (X right, Y down, Z forward). Store extrinsics
  as `T_world_cam` (camera-to-world, 4x4, row-major).
- Rotations: quaternions scalar-first `wxyz` (MuJoCo convention) in all stored
  artifacts. SciPy is `xyzw`: conversions happen only in `schema/rotations.py`
  with unit tests. No exceptions.
- Canonical timeline: 30 Hz. Keep original timestamps; resampled values carry an
  `interpolated` flag.
- Human model: SMPL-X neutral, 10 betas stored (16 accepted), axis-angle pose.
  Hands additionally exported as SE(3) for 25 joints per hand in the EgoDex
  convention.
- Files: tabular data in Parquet; per-frame dense arrays in NPZ; depth as 16-bit
  PNG in millimeters with scale recorded in `camera.json`; meshes as GLB; video
  as MP4 (h264, yuv420p, 30 fps).
- Every kinematic Parquet table has, per logical quantity: value columns, `conf`
  (float 0-1), `valid` (bool), `source`
  (enum: `captured | estimated | triangulated | fused | synthesized`).
- IDs: `episode_id = {source_id}_{clip_idx:06d}`. All artifacts live under
  `workspaces/{episode_id}/`.
