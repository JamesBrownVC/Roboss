# Per-stage tool environments

Every third-party research tool runs in its OWN isolated environment
(master prompt section 1: isolation over unification). These specs are
consumed on the Ubuntu 22.04+ / CUDA 12.x production host, never on the
Windows dev harness (which runs everything in `mode=synthetic`).

| env dir          | pipeline stage     | tool                      | spec kind  | license (mirror of config/licensing.yaml)          |
|------------------|--------------------|---------------------------|------------|-----------------------------------------------------|
| `vipe/`          | geometry           | ViPE (nv-tlabs)           | pixi.toml  | Apache-2.0 code; weights: verify                    |
| `gvhmr/`         | human_body         | GVHMR (zju3dv)            | pixi.toml  | research-only; commercial: verify                   |
| `wilor/`         | hands              | WiLoR                     | pixi.toml  | CC-BY-NC weights; commercial: no                    |
| `foundationpose/`| objects            | FoundationPose (NVlabs)   | Dockerfile | NVIDIA non-commercial; commercial: no               |
| `semantics/`     | semantics          | Qwen2.5-VL via vLLM       | pixi.toml  | Apache-2.0 (model + server); commercial: yes        |
| `gmr/`           | retarget (H1)      | GMR (YanjieZe)            | pixi.toml  | MIT; commercial: yes                                |
| `mink/`          | retarget (H2/H3)   | mink (kevinzakka)         | pixi.toml  | Apache-2.0; commercial: yes                         |
| `isaaclab/`      | physics Tier-2     | Isaac Lab + Isaac Sim     | Dockerfile | BSD-3 code / NVIDIA Omniverse EULA runtime          |

Rules that apply to every env in this directory:

1. **Pin everything.** Each spec carries a `PINNED_COMMIT` placeholder that the
   operator MUST replace with a real commit hash before production runs. The
   orchestrator records the commit and the sha256 of every weights file in the
   stage manifest.
2. **No weights are fetched automatically.** Every download step is commented
   out. Weights that require registration or license acceptance (SMPL-X, MANO,
   GVHMR checkpoints, WiLoR checkpoints, FoundationPose weights) must be
   reviewed and placed by the operator. SMPL-X / MANO are NEVER scraped; see
   `assets/body_models/README.md`.
3. **README wins on installation.** Where these specs disagree with the pinned
   repo's README on installation details, the README wins (master prompt,
   preamble). Where a README disagrees with the V2R interchange contract, the
   contract wins.
4. **Dockerfiles are reserved for fragile native builds** (FoundationPose,
   Isaac Lab), everything else uses pixi. The orchestrator invokes tools via
   `src/v2r/stages/base.py::run_tool` (`micromamba run -n {env} ...` or a full
   `docker run ...` command line).
5. `licensing.permissive_only=true` (config/licensing.yaml) disables the
   non-permissive tools entirely and switches the pipeline to the documented
   fallbacks with degraded confidence. See each NOTES.md.
