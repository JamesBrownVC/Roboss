# Semantics (Qwen2.5-VL via vLLM)

- **Purpose**: subtask segmentation labels, episode captions (short / medium /
  long), scene tags. Segmentation BOUNDARIES come from changepoint detection
  on hand aperture + contact transitions (the human-video analog of
  gripper-transition segmentation); the VLM only labels each segment.
- **Source**: model `Qwen/Qwen2.5-VL-7B-Instruct` (7B minimum; larger if VRAM
  allows), served by vLLM. Pin the HF revision via `PINNED_COMMIT`.
- **License**: Qwen2.5-VL weights Apache-2.0, vLLM Apache-2.0
  (`config/licensing.yaml`: `commercial: yes`). Note the master prompt's
  warning that SOME VLM checkpoints are non-commercial -- if the operator
  swaps the model, re-run the license audit.

## Contract

Writes `semantics/segments.json` (models.SegmentsFile: start/end seconds,
`skill` MUST be one of `config/verbs.yaml` -- anything else is a schema
violation), `semantics/captions.json` (models.Captions, three lengths),
`semantics/scene_tags.json` (models.SceneTags: scene_type, lighting,
clutter 1-5, surfaces).

## Gotchas

- Temperature 0 and JSON-schema-constrained outputs (vLLM `guided_json`),
  always. Free-text VLM output is never parsed with regexes and hoped for.
- The verb vocabulary is CLOSED (`config/verbs.yaml`). The wrapper passes the
  list in the schema enum; a response outside it is a stage failure, not a
  new verb.
- All semantics are auto-generated; human QA happens only on samples
  (master prompt 6.G). These are estimates -- `source=estimated` (or
  `synthesized` in synthetic mode), never ground truth.
- VRAM: 7B + 16k context + images fits in ~20 GB; on a 12 GB card reduce
  `--max-model-len` and image count per prompt.
