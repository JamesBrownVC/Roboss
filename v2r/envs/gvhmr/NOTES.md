# GVHMR (human_body stage)

- **Purpose**: world-grounded SMPL-X body motion from monocular video
  (betas, axis-angle pose, global_orient, transl per frame). Primary body
  tracker; fallbacks are WHAM and TRAM.
- **Source**: https://github.com/zju3dv/GVHMR (pin via `PINNED_COMMIT`).
- **License**: research-only (check the repo at the pinned commit).
  `config/licensing.yaml`: `commercial: verify`. GVHMR checkpoints are
  downloaded manually by the operator after accepting the license.
- **Body models**: SMPL-X (and MANO for the hands stage) require registration
  at the MPI sites and are NEVER scraped. The operator places them in
  `assets/body_models/`; the `link-body-models` task symlinks them into the
  tool layout. See `assets/body_models/README.md`.

## Contract

Writes `human/smplx.npz` with keys: `t (T,)`, `betas (10,)`,
`global_orient (T,3)` axis-angle, `body_pose (T,63)`, `transl (T,3)`,
`joints_world (T,22,3)` ordered as `schema.io.SMPLX_MAIN_JOINTS`, `conf (T,)`,
`valid (T,)` bool, `source` (0-d str). Plus `human/fusion_report.json`
(models.FusionReport).

Gate (config/qa.yaml): mean 2D reprojection error of SMPL-X joints against a
2D detector (Sapiens or ViTPose, run in THIS stage env) <= `max_reproj_px`;
joint acceleration (jitter) <= `max_jitter_m_s2`.

## Gotchas

- **World-frame harmonization is critical**: GVHMR's world frame is NOT
  ViPE's. The stage wrapper solves a similarity transform (Umeyama,
  `schema/alignment.py`) between GVHMR's camera trajectory and ViPE's
  `poses.parquet`, applies it to the body motion, writes residuals into
  `human/fusion_report.json`, and folds the residual into `conf`. Never ship
  un-harmonized body motion.
- GVHMR expects a moving-camera mode (DPVO) for handheld video; the static
  simple mode drifts. The relative-pose input can also come from ViPE
  (already computed) to skip DPVO.
- 10 betas are stored (16 accepted at input, truncated with a note).
- Sapiens weights are CC-BY-NC; under `licensing.permissive_only=true` the 2D
  gate detector must be ViTPose (Apache-2.0), per
  `config/licensing.yaml: permissive_fallbacks`.
