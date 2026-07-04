# WiLoR (hands stage)

- **Purpose**: per-frame MANO hand pose + shape from RGB crops; the hands
  stage fuses MANO into SMPL-X at the wrists and exports 25 SE(3) joints per
  hand in the EgoDex convention. Alternatives: HaMeR; egocentric world-space
  option: HaWoR.
- **Source**: https://github.com/rolpotamias/WiLoR (pin via `PINNED_COMMIT`).
- **License**: code permissive-ish, WEIGHTS are CC-BY-NC.
  `config/licensing.yaml`: `commercial: no`. Under `permissive_only=true` the
  pipeline substitutes HaMeR (verify weights license) or 2D-only hands with
  `conf_multiplier: 0.8`.
- **MANO**: registration-gated at mano.is.tue.mpg.de, NEVER scraped; the
  operator places it in `assets/body_models/` (see that README).

## Contract

Writes `human/hands.parquet` in `schema.io.HANDS_COLUMNS` long format:
one row per (frame, hand, joint), joints ordered per
`schema.io.EGODEX_HAND_JOINTS` (25 per hand: wrist + 4 thumb + 5 x 4
fingers), `hand in {left,right}`, quaternions wxyz, at the 30 Hz canonical
timeline.

Gate (config/qa.yaml): mean conf >= 0.30, invalid ratio <= 0.60.

## Gotchas

- Wrist fusion: replace the SMPL-X hand articulation with MANO and solve ONE
  fixed wrist offset per episode (not per frame); per-frame offsets hide
  tracker error inside the skeleton.
- Confidence = detector score x temporal-consistency score. Frames where a
  hand is occluded get `valid=false`; interpolation is allowed only with the
  `interpolated` flag and gaps capped at `pipeline.max_interp_gap_s` (0.34 s).
  Never fabricate values across long occlusions.
- WiLoR's detector runs per frame; left/right assignment can flip on
  ambiguous crops -- the wrapper enforces temporal identity consistency
  before export.
