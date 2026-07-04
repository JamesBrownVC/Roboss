# ViPE stage environment (Geometry / Stage B)

**Primary tool:** [ViPE](https://github.com/nv-tlabs/vipe) — camera intrinsics, `T_world_cam`, dense near-metric depth.

**Pinned commit:** `8f3c2a1b9d0e4f7a6c5b8d9e0f1a2b3c4d5e6f7` (replace with actual pin when upgrading)

**Alternatives (document only):** MegaSaM, VGGT, MASt3R-SLAM, COLMAP, Depth Anything

## Host requirements

- Ubuntu 22.04+, CUDA 12.x, NVIDIA GPU (24 GB recommended)
- micromamba or pixi

## Install

```bash
micromamba create -n vipe python=3.10 -y
micromamba activate vipe
git clone https://github.com/nv-tlabs/vipe.git
cd vipe && git checkout 8f3c2a1b9d0e4f7a6c5b8d9e0f1a2b3c4d5e6f7
# Follow ViPE README for torch/CUDA + model weights (operator-provided)
pip install -e .
```

## Invocation (from V2R orchestrator)

```bash
micromamba run -n vipe python envs/vipe/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --video workspaces/{episode_id}/raw/video.mp4
```

## Outputs (contract)

Writes `geometry/camera.json`, `geometry/poses.parquet`, `geometry/depth/*.png`, `geometry/scene.ply`.

## Switch synthetic → real

Set in `config/pipeline.yaml`:

```yaml
stages:
  geometry: {enabled: true, mode: real, env: vipe}
```

Or: `v2r run --episode ... --mode real` (overrides all stages).
