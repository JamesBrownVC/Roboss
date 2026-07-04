# GMR (retarget stage, H1 humanoid_wholebody adapter)

- **Purpose**: real-time whole-body retargeting of SMPL-X motion to 17+
  humanoids (Unitree G1/H1, Booster T1, Fourier GR3, Talos, ...). Reference
  implementation for the `humanoid_wholebody` class in `config/robots.yaml`.
- **Source**: https://github.com/YanjieZe/GMR (pin via `PINNED_COMMIT`).
- **License**: MIT (`config/licensing.yaml`: `commercial: yes`).

## Contract

Writes `retargets/{robot}/qpos.parquet` with columns: `t, frame, root_px,
root_py, root_pz, root_qw, root_qx, root_qy, root_qz`, one column per dof
name from `config/robots.yaml`, then `conf, valid, source, retarget_method,
retarget_version, provenance`. A CSV mirror (`qpos.csv`) is written for
BeyondMimic-style consumers. Plus `mapping.json` (models.RetargetMapping,
`provenance="kinematic-retarget"`) and `ee.parquet`
(`schema.io.EE_COLUMNS`).

## Gotchas

- **The monocular path is exactly this pipeline.** GMR documents a monocular
  route via GVHMR -- that is stage C. Feed GMR the HARMONIZED, world-frame,
  scale-corrected SMPL-X from `human/smplx.npz`, never raw GVHMR output;
  otherwise the robot walks in GVHMR's frame, not the episode world frame.
- Root pose is stored quaternion-first wxyz (MuJoCo convention) like every
  other artifact; GMR is MuJoCo-native so no reorder is needed, but the
  wrapper still routes any conversion through `schema/rotations.py`.
- GMR runs at the robot's own rate; the pipeline resamples to the 30 Hz
  canonical timeline for storage (control_rate_hz is in `config/robots.yaml`
  for consumers that re-interpolate).
- Requires operator-provided SMPL-X models and robot MJCF assets
  (`assets/body_models/`, `assets/robots/` -- see those READMEs).
- Every retarget is an ESTIMATE derived from estimated human motion:
  conf/valid propagate from the human stages and are never set to 1.0.
