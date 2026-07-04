# agents — Scenario Contract Agent

The **user → video generation** half of the Synthetic Action Dataset
Compiler. Input: a free-form user intention. Output: a *contract-based*
generation bundle — not a prompt, but a controlled scene specification.

```
user intention
      │
      ▼
 1. Intent Parser        Gemini, structured JSON → domain, actors, task, risk
 2. Contract Builder     Gemini → world_contract (immutable canvas:
                         entities + appearance locks, materials, layout,
                         lighting, style) + variation_policy (what may
                         change: camera angles, event types, deltas, timing)
 3. Scenario Planner     Gemini → N variations INSIDE the contract
 4. Validator            pure Python, no LLM: identity/camera/event/action
                         contracts checked deterministically; failures are
                         fed back to the model (repair loop), then dropped
 5. Prompt Compiler      pure Python: the video prompt is COMPILED from the
                         contract, so every prompt repeats the same
                         appearance locks verbatim + a consistency clause
 6. Canvas anchor        Gemini image model → one canonical scene image
    (optional)           shared by the whole series
 7. Start frames         canvas image + edit instruction → per-scenario
    (optional)           first frame for image-to-video generation
```

## Why a contract, not a prompt

Video models drift: the box changes shape, the shelf moves, the robot
changes color between generations. Cross-video consistency is held by
four anchors, each produced by this package:

| anchor | artifact | holds |
|---|---|---|
| semantic | `contract.json` (`world_contract`, `object_registry`, `scene_registry`, `variation_policy`) | identity, materials, layout, style, what may vary |
| textual | compiled `video_prompt` per scenario | same appearance wording verbatim in every prompt |
| visual | `canvas.png`, reference assets, `frames/<id>_start.png` | same look; start frames all derive from one canonical world |
| trust | `verifier_packets/<id>.json` | post-generation compliance check with object/scene identity checks |

Immutable (locked by the contract): entity ids, appearance, materials,
environment layout, lighting, style. Mutable (variation policy): camera
angle, event type, timing, risk severity, small position deltas, action
timeline.

The advanced consistency layer is registry-based:

- `object_registry`: one identity anchor per locked entity, including shape,
  material, color, scale, surface details, required reference views and
  negative drift examples.
- `scene_registry`: one background/layout anchor for the series, including
  spatial map, lighting, reference views and forbidden background drift.
- `reference_assets`: the visual assets the generator should create or reuse
  (`canvas.png`, object views, scene plates, start frames, masks/depth maps).
- `identity_checks`: post-generation checks for object crop similarity,
  background/layout similarity and VLM contract compliance.

## Usage

Needs `GEMINI_API_KEY` (same credential gate 2 of the verifier uses).

```powershell
python -m agents "Generate rare safety scenarios where a humanoid robot carries a box in a warehouse and a human worker behaves unpredictably" `
    --count 5 --out runs\warehouse_v1 --start-frames
```

Output bundle:

```
runs\warehouse_v1\
  intent.json            structured intent
  contract.json          world_contract + object/scene registries +
                         reference assets + identity checks + policy
  scenarios.json         N validated scenarios: camera, event, timeline,
                         expected_labels, keyframes, compiled video_prompt
  canvas.png             canonical canvas anchor (with --canvas/--start-frames)
  frames\sc_*_start.png  per-scenario start frames for image-to-video
  verifier_packets\*.json  one packet per scenario for the verification gate
  bundle.json            manifest
```

## Closing the loop with the verifier

Each scenario ships a `verifier_packet` in the exact format the rejection
gate consumes, so after the video model renders `sc_001.mp4`:

```powershell
python -m verifier sc_001.mp4 --scenario runs\warehouse_v1\verifier_packets\sc_001_human_slip.json --gate2
```

planned scenario → generated video → contract compliance + physics check →
only compliant videos enter the dataset.

## Tests

The validator and compiler are pure functions over dicts — tested without
any API in `tests/test_agents.py`:

```powershell
python -m pytest tests -q
```

## Config

Models and knobs in [config.py](config.py): `gemini-3.5-flash` for text
stages (same family as the verifier's gate 2), `gemini-3.5-flash-image`
for the visual anchors (override with `--image-model`), temperature 0.3
for planning stages vs 0.9 for scenario diversity, 2 repair rounds.
