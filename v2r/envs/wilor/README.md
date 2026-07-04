# WiLoR / HaMeR stage environment (Hands / Stage D)

**Primary:** [WiLoR](https://github.com/rolpotamias/WiLoR) (MANO)  
**Alternative:** [HaMeR](https://github.com/geopavlakos/hamer)  
**Egocentric world hands:** HaWoR

**Pinned commit (WiLoR):** `b2c3d4e5f6789012345678901234567890abcde1`

## Prerequisites

**MANO** models in `assets/body_models/mano/` (MPI registration).

## Install

```bash
micromamba create -n wilor python=3.10 -y
micromamba activate wilor
git clone https://github.com/rolpotamias/WiLoR.git
cd WiLoR && git checkout b2c3d4e5f6789012345678901234567890abcde1
pip install -e .
```

## Invocation

```bash
micromamba run -n wilor python envs/wilor/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --mano assets/body_models/mano \
  --format egodex25
```

Exports EgoDex 25-joint SE(3) to `human/hands.parquet`.

## Switch synthetic → real

```yaml
stages:
  hands: {enabled: true, mode: real, env: wilor}
```
