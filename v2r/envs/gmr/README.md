# GMR stage environment (Retarget / Stage H — humanoids)

**Primary (humanoid_wholebody):** [GMR](https://github.com/YanjieZe/GMR) — GVHMR→GMR monocular path

**Manipulators:** mink (differential IK) — separate env recommended  
**Quadruped:** command-abstraction adapter in orchestrator

**Pinned commit (GMR):** `d4e5f6789012345678901234567890abcdef1234`

## Install

```bash
micromamba create -n gmr python=3.10 -y
micromamba activate gmr
git clone https://github.com/YanjieZe/GMR.git
cd GMR && git checkout d4e5f6789012345678901234567890abcdef1234
pip install -e .
```

## Invocation

```bash
micromamba run -n gmr python envs/gmr/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --robot g1 \
  --robot-class humanoid_wholebody \
  --smplx workspaces/{episode_id}/human/smplx.npz
```

## Switch synthetic → real

```yaml
stages:
  retarget: {enabled: true, mode: real, env: gmr}
```
