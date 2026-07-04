# FoundationPose (objects stage)

- **Purpose**: 6DoF object pose estimation + tracking given an object mesh.
  Mesh-free path: BundleSDF or SAM 3D reconstructs a coarse mesh first, then
  FoundationPose tracks. Detection/segmentation prompts come from
  Grounding DINO + SAM 2 (both permissive), prompted with the
  manipulated-object list from the semantics stage.
- **Source**: https://github.com/NVlabs/FoundationPose (pin via
  `PINNED_COMMIT` build arg).
- **License**: NVIDIA NON-COMMERCIAL (`config/licensing.yaml`:
  `commercial: no`). BundleSDF is also NVIDIA non-commercial.

## permissive_only fallback (do not skip)

When `licensing.permissive_only=true` this image is NOT run. The objects
stage instead uses mask + depth ICP tracking (Open3D, custom, in the
orchestrator env) and marks all tracks with degraded confidence
(`conf_multiplier: 0.6` from `config/licensing.yaml`). The substitution is
surfaced in `LICENSE_AUDIT.md` and in the objects manifest.

## Contract

Writes `objects/tracks.parquet` (`schema.io.TRACKS_COLUMNS`: one row per
(frame, object_id), wxyz quaternions, conf/valid/source),
`objects/masks/{object_id}/...`, `objects/meshes/{object_id}.glb`. Includes
collision-relevant untouched objects near the workspace (pose + coarse mesh).

Gate (config/qa.yaml): `min_track_conf` 0.30.

## Gotchas

- The build is the fragile part: custom CUDA extensions (`build_all.sh`),
  nvdiffrast, kaolin, eigen/boost pybind modules -- hence Docker, per master
  prompt section 5. Do not try to fold this into a pixi env.
- FoundationPose needs reasonable depth; feed ViPE depth (already
  near-metric, scale-corrected) rather than raw sensor-less guesses.
- Weights are license-gated: operator downloads after accepting the NVIDIA
  license; the image never fetches them (the download step is commented out
  in the Dockerfile).
- Track identity: object_id is assigned by the detection pass and must stay
  stable across the episode; re-detections that change identity are a gate
  failure, not something to paper over.
