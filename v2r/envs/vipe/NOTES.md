# ViPE (geometry stage)

- **Purpose**: the pipeline backbone. Per-frame camera intrinsics,
  `T_world_cam`, dense near-metric depth, and dynamic-object masks from
  unconstrained monocular video. Supports pinhole / fisheye / 360. Runs at
  roughly 3-5 FPS on one GPU.
- **Source**: https://github.com/nv-tlabs/vipe (pin via `PINNED_COMMIT`).
- **License**: Apache-2.0 for the code; the model checkpoints it uses
  (GroundingDINO/SAM masking, depth priors) must be verified individually.
  `config/licensing.yaml`: `commercial: verify`.

## Contract

Writes `geometry/` per conventions: `camera.json` (models.CameraInfo),
`poses.parquet` (schema.io.POSES_COLUMNS, `T_world_cam` camera-to-world,
row-major), `depth/{frame:06d}.png` (16-bit, millimeters by default,
`depth_scale` recorded in camera.json). Scene fusion (`scene.ply`,
`scene_mesh.glb`) happens in the ORCHESTRATOR env (Open3D), not here.

Gate (config/qa.yaml): tracked-frame ratio >= 0.90, median depth
coverage >= 0.60.

## Gotchas

- **Scale is NEAR-metric, not metric.** Do not trust it blindly. The pipeline
  refines per-episode scale later against fused SMPL-X body height (stage C)
  or a known-size object, records `scale_source` and `scale_correction` in
  `camera.json`, and never overwrites the raw estimate silently.
- ViPE masks dynamic objects (GroundingDINO + SAM) so the solve locks onto the
  static scene; keep that enabled for hand/object-heavy manipulation clips.
- Depth PNGs may be stored at reduced resolution (`depth_width`/`depth_height`
  in camera.json) to keep episodes small; consumers must check those fields.
- Alternatives if ViPE fails on a clip class: MegaSaM, VGGT, MASt3R-SLAM;
  COLMAP for static multi-view calibration sessions.
