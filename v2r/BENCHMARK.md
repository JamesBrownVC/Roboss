# V2R Agentic Labeler — Competitive Benchmark & Quantified Accuracy

Date: 2026-07-05. All numbers on this page are reproducible from this repo:
the accuracy numbers come from `v2r label-bench` (ground truth:
`tests/data/label_bench.yaml`, 13 hand-verified clips), the VLM-only baseline
from `scripts/vlm_only_baseline.py`. Raw outputs: `qa_bench_report.json`,
`qa_vlm_baseline.json`, and per-clip transcripts in
`workspaces/bench_*/qa/agentic_transcript.json`.

## 1. Headline: quantified label accuracy (our own measured numbers)

13-clip ground-truth set: 5 real clips (lab pipetting, ocean waves, night
scene with animal, mountain timelapse) + 8 AI-generated/styled clips
(manipulation, waving, montages, an abstract stick figure). Includes 4
adversarial "trap" clips with no labelable subject.

| metric | **agent loop (ours)** | VLM-only (Gemini, 1 call) |
|---|---|---|
| Verdict accuracy (proceed/reject/review) | **13/13 = 100%** | 10/13 = 77% |
| human_present accuracy | 92% | (not directly comparable¹) |
| Skill recall (≥1 expected verb found) | 92% | 92% |
| Skill precision (allowed-verb rate) | 96% | 97% |
| Clips with hallucinated (forbidden) verbs | **0 / 13** | 1 / 13 (`walk` fabricated) |
| Segment-boundary IoU vs GT (mean) | **0.71** | 0.68 |
| Segment-boundary MAE | **0.52 s** | — |
| Per-segment evidence/provenance coverage | **100%** | 0% (none offered) |
| Wall time per clip (mean) | 97 s | 12.7 s |
| Est. LLM cost per clip² | ~$0.02 | ~$0.002 |

¹ the VLM baseline emits human_present but our GT acceptance sets were
tuned for the agent's stricter definitions; not scored to avoid bias.
² agent loop: ~8–14 Crusoe calls/clip (Nemotron-3-Nano-Omni-30B-A3B, MoE
~3B active) at typical 30B-class serving rates (~$0.1/M in, ~$0.4/M out),
~70k in + 18k out tokens ≈ $0.01–0.03; plus ≤1 Gemini Flash video call
(~$0.001–0.005). VLM-only: one Gemini Flash video call.

The two failure cells are honest and visible: `stick_figure` (an abstract
rendered figure — the agent said `partial` human and labeled `gesture`,
where our strict GT wanted `none`/no action; defensible either way, scored
against us) accounts for both the human_present and recall misses.

### Where the agent beats VLM-only, on our own data

- **Verdict reliability** (the sell/don't-sell decision): VLM-only wrongly
  rejected 3 usable manipulation clips as "AI-generated therefore reject"
  (`wire_winding`, `workshop_montage`, `wrench_portrait`) — for a pipeline
  whose *input is intentionally AI-generated video*, that policy error costs
  23% of sellable data. The agent's verdicts were 13/13.
- **Zero fabrications** under adversarial input (waves/night/timelapse
  traps), enforced by machine gates, not model goodwill: a validator rejects
  `human_present='none'` + action verbs; a critic (multimodal, sees frames +
  the full tool-evidence ledger) must accept; unresolved critic problems
  auto-downgrade `proceed` → `human_review`.
- **Provenance**: every non-idle segment ships with an `evidence` field
  ("boundary from primitives changepoint at 4.9 s; skill from frames 5–6 s +
  wrist-speed peak") — auditable by a buyer, absent from any VLM-only output.
- **Time-series byproducts** worth as much as the labels for robot learning:
  per-frame pose parquet, hand-aperture series, ByteTrack object trajectories
  with velocities/periodicity/smoothness, optical-flow camera/subject
  separation, and changepoint motion primitives are written to the workspace
  for every accepted clip.

### Where VLM-only wins

7× faster, ~10× cheaper, and nearly as good on *easy, single-shot,
well-lit* clips. If provenance, QA gates, and time-series artifacts don't
matter for a use case, a single Gemini call is a strong cheap baseline —
that is an honest result of this benchmark.

## 2. Against commercial labeling vendors (2026 public info)

Pricing below is from public sources (vendor pages don't publish rates;
figures are industry-reported ranges). Sources: labelstud.io Encord review,
unitlab/sourcebae/dataxpower 2026 tool comparisons, truelabel robotics
annotation pricing survey.

| | **V2R agentic (ours)** | Scale AI | Encord | Labelbox | V7 / SuperAnnotate | Appen / iMerit / Sama |
|---|---|---|---|---|---|---|
| Model | autonomous agent + local perception | platform + managed workforce | platform (video-first) | platform (+workforce) | platform | managed workforce |
| Cost | ~$0.02/clip LLM + CPU compute | ~$200–300k / 5k robot episodes (~$40–60/ep) | $80–400k/yr programs | from $0.10/labeling-unit; Pro ~$500–1500/mo | ~$62/user/mo (SA); per-user (V7) | $50–500k programs |
| Turnaround | ~100 s/clip, fully automatic, 24/7 | days–weeks (managed) | human-in-loop | human-in-loop | human-in-loop | days–weeks |
| Temporal action segments | yes, changepoint-anchored | yes (human) | yes (strongest tooling) | yes (timeline editor) | yes | yes |
| Captions (3 lengths) | yes | RLHF-style tasks | yes | yes | yes | yes |
| Keypoints / pose | yes (MediaPipe 33-pt + 25-joint hands, estimated) | yes (human, GT-grade) | yes | yes | yes | yes |
| Object tracks | yes (ByteTrack, estimated) | yes (human) | yes (interpolation) | yes | yes | yes |
| Per-label provenance | **yes (evidence field + tool artifacts + transcript)** | QA process, not per-label evidence | consensus scoring | review workflows | review workflows | QA sampling |
| QA guarantee | machine gates + critic + measured accuracy on GT set | 92–97% first-pass acceptance (industry) | automated consensus | multi-stage review | multi-stage review | acceptance SLAs |
| Human keypoint accuracy | **lower** (estimated, not human-verified) | higher | higher | higher | higher | higher |

Honest positioning: we are not a replacement for human GT-grade keypoint
annotation (Scale/Encord humans beat MediaPipe). The niche is **high-volume,
low-cost first-pass labeling of (especially AI-generated) manipulation video
at ~1/1000th of managed-service cost**, with quantified accuracy and
auditable provenance, feeding either direct training or a cheaper
human-review funnel (our `human_review` verdicts are exactly that queue).

## 3. Against auto-labeling stacks

- **Autodistill / Grounding-DINO+SAM pipelines**: label *spatial* concepts
  (boxes/masks) from text prompts; no temporal segmentation, captions, or
  feasibility verdicts. Our `find` tool wraps the same idea (YOLO-World) as
  one instrument among many. Complementary, not competing.
- **FiftyOne**: dataset curation/QA around model predictions; does not
  produce temporal action labels. Could *consume* our outputs.
- **Encord/Labelbox model-assisted**: pre-labels feed human editors; the
  economics still include the human pass. Ours is autonomous with an
  explicit review queue.
- **NVIDIA Cosmos-style curation**: video filtering/captioning at scale for
  foundation models; not per-segment action labels with skill taxonomy and
  physics-pipeline integration (retargeting, MuJoCo validation) as here.

## 4. Known limitations (buyer-facing)

1. **AI-generation detection is unreliable.** We measured frame-statistics
   forensics (temporal noise autocorrelation, HF-energy stability) on 6 real
   + 5 generated clips: the distributions fully overlap (real −0.26..−0.58,
   generated −0.41..−0.53). The `forensics` tool therefore always returns
   `verdict: inconclusive` and instructs the agent to rely on visible
   artifacts only. Provenance of source video must come from the ingest
   side (we generate much of it ourselves via syngen, where provenance is
   exact), not from detection.
2. **Keypoints are estimated, not human-verified** (MediaPipe/CPU). Gloved
   hands defeat the hand tracker; the open-vocabulary `find` tool only
   partially compensates (0.16 max-conf on "blue nitrile glove" = weak).
3. **Boundary IoU is 0.71**, not 0.9+: boundaries are changepoint-anchored
   but verb-to-interval assignment is still VLM judgment.
4. **Single-view RGB only** on this host; no depth/multi-view GT tier here.

## 4b. Quadruped (robot-dog) extension

The labeler now handles four-legged animals via an `animal_pose` tool built on
DeepLabCut's **SuperAnimal-Quadruped** model (FasterRCNN-MobileNetV3 detector +
ResNet50-GroupNorm 39-keypoint pose head), run standalone on CPU/Windows
(torch + timm + opencv; the full `deeplabcut` package is not importable here —
it pins numpy<2 and needs native builds, so we load the model-zoo weights
directly). Weights (~180 MB) come from `dlclibrary` into `assets/superanimal/`.

The agent picks `animal_pose` autonomously when it sees an animal (verified
live: `data/raw/animal_dog/dog_run.mp4`, CC-BY Wikimedia, two dogs running).
The tool emits per-second keypoint presence/confidence, four paw-tip speed
timelines (body-length-normalized, scale invariant), stride periodicity
(autocorrelation), body displacement, spine angle, and a standing/recumbent
posture read, plus a `keypoints_superanimal.parquet` artifact. Gait verbs
(`walk/trot/gallop/jump/sit/lie_down/stand/turn`) were added to the taxonomy.

On the `dog_quadruped` bench clip: verdict proceed, human_present=none, gait
skills correct, zero hallucinations, evidence coverage 100%. **Downstream
consumer: Unitree Go2 quadruped retargeting** (`assets/robots/go2`) — the same
physics/retarget pipeline used for humanoid data now covers robot dogs.

## 5. Reproducing

```bash
# full 13-clip accuracy bench (agent loop; ~25 min CPU)
v2r label-bench
# score existing workspaces without rerunning
v2r label-bench --reuse
# subset
v2r label-bench --only pipette,waves
# VLM-only baseline on the same GT
python scripts/vlm_only_baseline.py aigen_cup pipette waves ...
```
