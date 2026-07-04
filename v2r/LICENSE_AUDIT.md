# V2R License Audit

Generated: 2026-07-04 22:46 UTC
**permissive_only**: `False`

## Third-party components

| Name | Kind | Ref | License | Commercial |
|------|------|-----|---------|------------|
| ViPE | repo | github.com/nv-tlabs/vipe | Apache-2.0 (code); check model weights | verify |
| GVHMR | repo | github.com/zju3dv/GVHMR | research-only (check repo) | verify |
| WHAM | repo | github.com/yohanshin/WHAM | research-only (check repo) | verify |
| WiLoR | repo | github.com/rolpotamias/WiLoR | CC-BY-NC (weights) | False |
| HaMeR | repo | github.com/geopavlakos/hamer | MIT (code); weights research-only | verify |
| GroundingDINO | repo | github.com/IDEA-Research/GroundingDINO | Apache-2.0 | True |
| SAM2 | repo | github.com/facebookresearch/sam2 | Apache-2.0 | True |
| FoundationPose | repo | github.com/NVlabs/FoundationPose | NVIDIA non-commercial | False |
| BundleSDF | repo | github.com/NVlabs/BundleSDF | NVIDIA non-commercial | False |
| Qwen2.5-VL | weights | huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Apache-2.0 | True |
| GMR | repo | github.com/YanjieZe/GMR | MIT | True |
| mink | repo | github.com/kevinzakka/mink | Apache-2.0 | True |
| LeRobot | repo | github.com/huggingface/lerobot | Apache-2.0 | True |
| MuJoCo | repo | github.com/google-deepmind/mujoco | Apache-2.0 | True |
| rerun | repo | github.com/rerun-io/rerun | MIT/Apache-2.0 | True |
| PySceneDetect | repo | github.com/Breakthrough/PySceneDetect | BSD-3 | True |
| EgoBlur | repo | github.com/facebookresearch/EgoBlur | Apache-2.0 | True |
| SMPL-X | weights | smpl-x.is.tue.mpg.de | MPI research license; commercial via Meshcapade | verify |
| MANO | weights | mano.is.tue.mpg.de | MPI research license; commercial via Meshcapade | verify |
| Sapiens | repo | github.com/facebookresearch/sapiens | CC-BY-NC (weights) | False |
| ViTPose | repo | github.com/ViTAE-Transformer/ViTPose | Apache-2.0 | True |

## Permissive-only fallbacks

- **FoundationPose** → mask + depth ICP tracking (Open3D, custom) (conf×0.6)
- **BundleSDF** → SAM 3D coarse mesh + ICP (conf×0.6)
- **WiLoR** → HaMeR (verify weights license) or 2D-only hands (conf×0.8)
- **Sapiens** → ViTPose (conf×0.9)

## Operator-provided assets (never scraped)

- SMPL-X: place registered models in `assets/body_models/`
- MANO: place registered models in `assets/body_models/mano/`
- Robot URDF/MJCF: `assets/robots/{name}/`

## Notes

FoundationPose and BundleSDF are NVIDIA non-commercial. Set `licensing.permissive_only: true` in config for Open3D ICP fallbacks.
