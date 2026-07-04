# GVHMR stage environment (Human body / Stage C)

**Primary tool:** [GVHMR](https://github.com/zju3dv/GVHMR) — world-grounded SMPL-X motion.

**Pinned commit:** `a1b2c3d4e5f6789012345678901234567890abcd`

**Fallbacks:** WHAM, TRAM | **2D QA:** Sapiens (ViTPose if `permissive_only`)

## Prerequisites

Operator must place **SMPL-X** models in `assets/body_models/` (MPI registration — never scraped).

## Install

```bash
micromamba create -n gvhmr python=3.10 -y
micromamba activate gvhmr
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR && git checkout a1b2c3d4e5f6789012345678901234567890abcd
pip install -e .
# Download GVHMR checkpoints per repo README
```

## Invocation

```bash
micromamba run -n gvhmr python envs/gvhmr/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --body-models assets/body_models \
  --align-vipe
```

Umeyama alignment to ViPE world frame is performed in-tool; writes `human/fusion_report.json`.

## Switch synthetic → real

```yaml
stages:
  human_body: {enabled: true, mode: real, env: gvhmr}
```
