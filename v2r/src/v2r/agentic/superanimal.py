"""Standalone SuperAnimal-Quadruped pose inference (CPU/Windows friendly).

Runs the DeepLabCut Model-Zoo SuperAnimal-Quadruped top-down pose estimator
(FasterRCNN-MobileNetV3 detector + ResNet50-GroupNorm heatmap pose head)
without importing the full ``deeplabcut`` package, which pins numpy<2 and pulls
heavy/native deps that do not build on this machine.

Weights come from ``dlclibrary`` (Hugging Face model zoo) and live under
``assets/superanimal/``:

    superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt   (~76 MB)
    superanimal_quadruped_resnet_50.pt                          (~103 MB)

The pose backbone is a GroupNorm ResNet-50 (timm ``resnet50_gn``), which is why
the checkpoint carries no BatchNorm running statistics: GroupNorm is
deterministic and needs none. Decode geometry (stride 8, locref_std 7.2801,
raw-heatmap scores clipped to [0,1]) matches DeepLabCut's ``HeatmapPredictor``.

Only ``torch``, ``torchvision``, ``timm`` and ``opencv`` are required, all of
which the ``ml`` env already has (CPU torch 2.x).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# 39 SuperAnimal-Quadruped body parts (DeepLabCut project config order)
QUADRUPED_KEYPOINTS = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base",
    "right_antler_end", "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base", "neck_end",
    "throat_base", "throat_end", "back_base", "back_end", "back_middle",
    "tail_base", "tail_end", "front_left_thai", "front_left_knee",
    "front_left_paw", "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai", "back_left_knee",
    "back_right_knee", "back_right_paw", "belly_bottom", "body_middle_right",
    "body_middle_left",
]
KP_INDEX = {name: i for i, name in enumerate(QUADRUPED_KEYPOINTS)}
PAW_KEYPOINTS = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
SPINE_KEYPOINTS = ["neck_base", "back_base", "back_middle", "tail_base"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_POSE_INPUT = 256          # square crop side fed to the pose net
_LOCREF_STD = 7.2801       # DeepLabCut location-refinement scale
_DECONV_STRIDE = 2         # single transpose-conv upsample in the head


def model_paths(assets_root: Path) -> tuple[Path, Path]:
    d = Path(assets_root) / "superanimal"
    return (
        d / "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt",
        d / "superanimal_quadruped_resnet_50.pt",
    )


def models_available(assets_root: Path) -> bool:
    det, pose = model_paths(assets_root)
    return det.is_file() and pose.is_file()


def _build_pose_net(n_kpts: int = 39):
    import timm
    import torch.nn as nn

    class _QuadrupedPoseNet(nn.Module):
        def __init__(self, nk: int):
            super().__init__()
            # GroupNorm ResNet-50, dilated to output_stride 16 (no running stats)
            self.model = timm.create_model(
                "resnet50_gn", output_stride=16, pretrained=False,
                num_classes=0, global_pool="",
            )
            self.heatmap = nn.ConvTranspose2d(
                2048, nk, 3, stride=_DECONV_STRIDE, padding=1, output_padding=1)
            self.locref = nn.ConvTranspose2d(
                2048, nk * 2, 3, stride=_DECONV_STRIDE, padding=1, output_padding=1)

        def forward(self, x):
            f = self.model.forward_features(x)
            return self.heatmap(f), self.locref(f)

    return _QuadrupedPoseNet(n_kpts)


@lru_cache(maxsize=2)
def _load_models(assets_root_str: str):
    """Load detector + pose net once per process (cached by assets root)."""
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    det_path, pose_path = model_paths(Path(assets_root_str))

    detector = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=2)
    det_sd = torch.load(det_path, map_location="cpu", weights_only=False)["model"]
    det_sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in det_sd.items()}
    detector.load_state_dict(det_sd, strict=False)
    detector.eval()

    pose = _build_pose_net(len(QUADRUPED_KEYPOINTS))
    pose_sd = torch.load(pose_path, map_location="cpu", weights_only=False)["model"]
    remapped = {}
    for k, v in pose_sd.items():
        if k.startswith("backbone.model."):
            remapped["model." + k[len("backbone.model."):]] = v
        elif "heatmap_head.deconv_layers.0." in k:
            remapped["heatmap." + k.split("deconv_layers.0.")[1]] = v
        elif "locref_head.deconv_layers.0." in k:
            remapped["locref." + k.split("deconv_layers.0.")[1]] = v
    pose.load_state_dict(remapped, strict=False)
    pose.eval()
    return detector, pose


def detect_animal(frame_bgr, assets_root: Path, min_score: float = 0.2):
    """Return the top animal bbox [x0,y0,x1,y1] and its score, or (None, 0.0).

    The SuperAnimal detector is a single-class (animal) FasterRCNN, so a
    detection above threshold is evidence a quadruped is present.
    """
    import cv2
    import torch

    detector, _ = _load_models(str(assets_root))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = detector([t])[0]
    if len(out["scores"]) == 0:
        return None, 0.0
    score = float(out["scores"][0])
    if score < min_score:
        return None, score
    return [float(v) for v in out["boxes"][0].tolist()], score


def pose_from_crop(frame_bgr, bbox, assets_root: Path, pad_frac: float = 0.2):
    """Run the pose net on a padded crop of ``bbox``; return (39, 3) array of
    (u, v, conf) in full-frame pixel coordinates."""
    import cv2
    import torch

    _, pose = _load_models(str(assets_root))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((len(QUADRUPED_KEYPOINTS), 3), dtype=np.float32)

    scale = _POSE_INPUT / max(ch, cw)
    nh, nw = int(round(ch * scale)), int(round(cw * scale))
    canvas = np.zeros((_POSE_INPUT, _POSE_INPUT, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(crop, (nw, nh))
    inp = (canvas.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    ti = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        hm, lr = pose(ti)
    hm = hm[0].numpy()
    lr = lr[0].numpy()
    nk, hh, hw = hm.shape
    stride = _POSE_INPUT / hh

    pts = np.zeros((nk, 3), dtype=np.float32)
    for c in range(nk):
        flat = int(hm[c].argmax())
        py, px = divmod(flat, hw)
        conf = float(np.clip(hm[c, py, px], 0.0, 1.0))  # config: apply_sigmoid=false
        ox = float(lr[2 * c, py, px]) * _LOCREF_STD
        oy = float(lr[2 * c + 1, py, px]) * _LOCREF_STD
        ix = px * stride + 0.5 * stride + ox
        iy = py * stride + 0.5 * stride + oy
        # undo pad-resize, then crop offset -> full frame pixels
        pts[c, 0] = x0 + ix / scale
        pts[c, 1] = y0 + iy / scale
        pts[c, 2] = conf
    return pts
