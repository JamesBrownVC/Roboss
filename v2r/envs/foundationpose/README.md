# FoundationPose stage environment (Objects / Stage E)

**Stack:** Grounding DINO + SAM 2 → FoundationPose (6DoF) | mesh-free: BundleSDF, SAM 3D

**Pinned commit (FoundationPose):** `c3d4e5f6789012345678901234567890abcdef12`

**License:** NVIDIA non-commercial. When `licensing.permissive_only: true`, V2R uses mask + depth ICP (Open3D).

## Install

```bash
micromamba create -n foundationpose python=3.10 -y
micromamba activate foundationpose
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose && git checkout c3d4e5f6789012345678901234567890abcdef12
# Follow README + Docker option for fragile builds
```

Also install Grounding DINO + SAM2 in this env or sibling envs per your ops layout.

## Invocation

```bash
micromamba run -n foundationpose python envs/foundationpose/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --permissive-only false
```

## Switch synthetic → real

```yaml
stages:
  objects: {enabled: true, mode: real, env: foundationpose}
```
