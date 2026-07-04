"""Geometry stage (master prompt 6.B, the backbone): per-frame intrinsics,
T_world_cam, dense near-metric depth, static scene reconstruction.

real mode: ViPE (github.com/nv-tlabs/vipe) in its isolated env.

    repo:   https://github.com/nv-tlabs/vipe
    commit: <PIN_ME_VIPE_COMMIT>            (record in envs/vipe/pixi.toml)
    env:    'vipe' (config/pipeline.yaml stages.geometry.env)
    cli:    vipe infer <video> --output <dir>   (README wins on install/CLI
            details at the pinned commit; this module wins on the interface)

    Expected raw output layout translated by ``_translate_vipe`` (VERIFY when
    pinning the commit; adjust the finder below, not the contract):
        {out}/{stem}/pose*.npz        ts (N,), (N,4,4) OpenCV cam-to-world
        {out}/{stem}/intrinsics*.npz  K (3,3) or fx/fy/cx/cy scalars
        {out}/{stem}/depth/*          per-frame metric depth (npz or 16-bit png)

    ViPE exposes no per-frame confidence at the CLI: translated poses carry a
    constant heuristic conf (noted in metrics) with source='estimated'. Scale
    is near-metric; scale_source='vipe_near_metric', refined later vs SMPL-X
    height (stage C) which updates scale_correction.

synthetic mode (any host, deterministic via base.rng_for): a smooth slow
camera orbit around the origin looking at the scene center, a plane+bump
depth field at quarter resolution, back-projected scene.ply, and a
table+floor scene_mesh.glb. Everything tagged source='synthesized'.

Gate (config/qa.yaml `geometry`): tracked_ratio >= min_tracked_ratio,
depth_coverage >= min_depth_coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from ..schema import io as sio
from ..schema.models import CameraInfo, StageStatus, VideoProbe
from ..schema.rotations import se3_from_quat_pos, se3_to_quat_pos, transform_points
from ..schema.timeline import canonical_timestamps, resample_linear, resample_quat_wxyz
from .base import (
    Stage,
    StageContext,
    StageResult,
    gate_from_thresholds,
    register_stage,
    rng_for,
    run_tool,
)

VIPE_REPO = "https://github.com/nv-tlabs/vipe"
VIPE_COMMIT = "PIN_ME_VIPE_COMMIT"
# Alternatives if ViPE underperforms (master prompt 6.B gotchas, not wired):
# MegaSaM, VGGT, MASt3R-SLAM; COLMAP for static multi-view calibration.

# Z-up world. The synthetic camera is a handheld eye-height shot framing a
# person zone ~3 m ahead: SCENE_CENTER is the look-at point (mid-body height
# at the person's distance) so a standing 1.75 m human fits the vertical fov.
SCENE_CENTER = np.array([0.0, 3.0, 0.9])
CAM_EYE_HEIGHT = 1.35


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _lookat_T_world_cam(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world SE(3) for OpenCV camera axes (X right, Y down,
    Z forward) at world positions `pos` (N,3) looking at `target` (3,)."""
    z = target[None, :] - pos
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    y_approx = np.broadcast_to(np.array([0.0, 0.0, -1.0]), z.shape)  # down
    x = np.cross(y_approx, z)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)  # x cross y = z (right-handed)
    R = np.stack([x, y, z], axis=-1)  # columns = camera axes in world
    T = np.zeros((len(pos), 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


def _backproject_world(
    depth: np.ndarray,
    cam: CameraInfo,
    T_world_cam: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample valid depth pixels (quarter-res grid) and lift them to world."""
    dh, dw = depth.shape
    sx, sy = dw / cam.width, dh / cam.height
    fx, fy = cam.fx * sx, cam.fy * sy
    cx, cy = cam.cx * sx, cam.cy * sy
    vv, uu = np.nonzero(depth > 0)
    if len(vv) == 0:
        return np.zeros((0, 3))
    sel = rng.choice(len(vv), size=min(n_points, len(vv)), replace=False)
    d = depth[vv[sel], uu[sel]]
    x = (uu[sel] + 0.5 - cx) * d / fx
    y = (vv[sel] + 0.5 - cy) * d / fy
    pts_cam = np.stack([x, y, d], axis=1)
    return transform_points(T_world_cam, pts_cam)


def _write_scene_mesh(path: Path) -> None:
    """Coarse static scene: table top near the scene center + floor plane."""
    table = trimesh.creation.box(extents=(1.2, 0.8, 0.04))
    table.apply_translation([SCENE_CENTER[0], SCENE_CENTER[1], SCENE_CENTER[2] - 0.02])
    floor = trimesh.creation.box(extents=(4.0, 4.0, 0.02))
    floor.apply_translation([0.0, SCENE_CENTER[1], -0.01])
    table.visual.face_colors = [160, 120, 80, 255]
    floor.visual.face_colors = [90, 90, 95, 255]
    scene = trimesh.Scene({"table_top": table, "floor": floor})
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


@register_stage
class Geometry(Stage):
    name = "geometry"

    def run(self, ctx: StageContext) -> StageResult:
        if ctx.mode == "real":
            return self._run_real(ctx)
        return self._run_synthetic(ctx)

    # ------------------------------------------------------------------
    # synthetic
    # ------------------------------------------------------------------
    def _run_synthetic(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        rng = rng_for(ws.episode_id, self.name)
        probe = sio.read_json_model(ws.probe_path, VideoProbe)
        hz = ctx.cfg.pipeline.canonical_hz
        t = canonical_timestamps(probe.duration_s, hz)
        n = len(t)
        w, h = probe.width, probe.height
        ctx.log(f"[geometry] synthesizing {n} frames at {hz:g} Hz ({w}x{h})")

        # camera.json ----------------------------------------------------
        cam = CameraInfo(
            model="pinhole",
            width=w,
            height=h,
            fx=0.92 * w,
            fy=0.92 * w,
            cx=w / 2.0,
            cy=h / 2.0,
            depth_scale=1000.0,
            depth_width=w // 4,
            depth_height=h // 4,
            scale_source="synthetic",
        )
        sio.write_json_model(ws.camera_json, cam)

        # trajectory: smooth slow orbit/pan around the origin -------------
        radius = 0.15 * (1.0 + rng.normal(0.0, 0.05))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        omega = rng.uniform(0.10, 0.25)  # rad/s: slow pan
        theta = phase + omega * t
        pos = np.stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                CAM_EYE_HEIGHT + 0.03 * np.sin(0.4 * t + phase),
            ],
            axis=1,
        )
        pos = pos + rng.normal(0.0, 0.0015, size=pos.shape)  # small noise
        T_world_cam = _lookat_T_world_cam(pos, SCENE_CENTER)

        conf = np.clip(0.9 + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
        valid = np.ones(n, dtype=bool)
        n_bad = max(1, int(round(0.02 * n))) if n > 10 else 0
        if n_bad:
            bad = rng.choice(n, size=n_bad, replace=False)
            valid[bad] = False
            conf[bad] = rng.uniform(0.05, 0.25, size=n_bad)
        tracked_ratio = float(valid.mean())

        frames = np.arange(n, dtype=np.int64)
        df = sio.poses_df(t, frames, T_world_cam, conf, valid, "synthesized")
        sio.write_table(df, ws.poses_parquet, required_columns=sio.POSES_COLUMNS)

        # depth: background wall just behind the person zone (~3 m, see
        # SCENE_CENTER) + gaussian bump for a nearer table object ----------
        dw, dh = cam.depth_width, cam.depth_height
        uu, vv = np.meshgrid(np.arange(dw), np.arange(dh))
        base = 3.05 + 0.25 * (vv / max(dh - 1, 1))
        bu = rng.uniform(0.40, 0.60) * dw
        bv = rng.uniform(0.50, 0.70) * dh
        sig = 0.06 * dw
        bump = 0.45 * np.exp(-((uu - bu) ** 2 + (vv - bv) ** 2) / (2.0 * sig**2))
        depth0 = base - bump
        bw = max(1, int(round(0.04 * dw)))
        bh = max(1, int(round(0.04 * dh)))
        mask = np.zeros((dh, dw), dtype=bool)
        mask[bh : dh - bh, bw : dw - bw] = True  # ~85% finite coverage
        depth0 = np.where(mask, depth0, 0.0)
        depth_coverage = float((depth0 > 0).mean())

        phase_d = rng.uniform(0.0, 2.0 * np.pi)
        ws.depth_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            scale_k = 1.0 + 0.008 * np.sin(2.0 * np.pi * 0.1 * t[k] + phase_d)
            sio.write_depth_png(ws.depth_frame(k), depth0 * scale_k, cam.depth_scale)
            if k and k % 300 == 0:
                ctx.log(f"[geometry] depth frames {k}/{n}")

        # scene.ply: back-projected keyframe depth ------------------------
        kf = np.unique(np.linspace(0, n - 1, 4).astype(int))
        per = max(1, 20000 // len(kf))
        pts, cols = [], []
        for k in kf:
            p = _backproject_world(depth0, cam, T_world_cam[k], per, rng)
            g = np.clip((p[:, 2] - p[:, 2].min()) / (np.ptp(p[:, 2]) + 1e-9), 0, 1)
            c = np.stack([200 * g + 40, 120 * np.ones_like(g), 220 * (1 - g) + 30], axis=1)
            pts.append(p)
            cols.append(c.astype(np.uint8))
        cloud = trimesh.PointCloud(np.concatenate(pts), colors=np.concatenate(cols))
        cloud.export(str(ws.scene_ply))

        # scene_mesh.glb ---------------------------------------------------
        _write_scene_mesh(ws.scene_mesh_glb)

        return self._gate_and_result(
            ctx,
            metrics={
                "tracked_ratio": tracked_ratio,
                "depth_coverage": depth_coverage,
                "n_frames": n,
            },
            tool="synthetic",
        )

    # ------------------------------------------------------------------
    # real: ViPE CLI in env 'vipe', then contract translation
    # ------------------------------------------------------------------
    def _run_real(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        env = ctx.cfg.stage(self.name).env or "vipe"
        out_root = ws.geometry_dir / "vipe_raw"
        out_root.mkdir(parents=True, exist_ok=True)
        ctx.log(f"[geometry] running ViPE in env '{env}'")
        proc = run_tool(
            ["vipe", "infer", str(ws.video_path), "--output", str(out_root)],
            env_name=env,
            timeout=7200,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise RuntimeError(
                f"ViPE failed (rc={proc.returncode}) in env {env!r} "
                f"at pinned commit {VIPE_COMMIT}: {tail}"
            )
        return self._translate_vipe(ctx, out_root)

    def _translate_vipe(self, ctx: StageContext, out_root: Path) -> StageResult:
        """Translate raw ViPE outputs into the contract artifacts.

        Layout assumptions documented in the module docstring; every mismatch
        raises RuntimeError naming the missing piece so the operator can fix
        the finder against the pinned commit's actual output tree.
        """
        ws = ctx.ws
        notes: list[str] = []
        probe = sio.read_json_model(ws.probe_path, VideoProbe)
        hz = ctx.cfg.pipeline.canonical_hz
        max_gap = ctx.cfg.pipeline.max_interp_gap_s
        t_dst = canonical_timestamps(probe.duration_s, hz)
        n = len(t_dst)

        base = next((p for p in sorted(out_root.iterdir()) if p.is_dir()), out_root)

        # poses ------------------------------------------------------------
        pose_file = self._find_file(base, ("pose", "poses", "camera"), ".npz")
        if pose_file is None:
            raise RuntimeError(f"no pose npz under {base}; verify ViPE layout")
        data = sio.read_npz(pose_file)
        T_src = next(
            (np.asarray(v, np.float64) for v in data.values()
             if getattr(v, "ndim", 0) == 3 and v.shape[-2:] == (4, 4)),
            None,
        )
        if T_src is None:
            raise RuntimeError(f"{pose_file} has no (N,4,4) pose array")
        t_src = next(
            (np.asarray(data[k], np.float64).reshape(-1)
             for k in ("ts", "t", "timestamps", "times")
             if k in data and data[k].size == len(T_src)),
            None,
        )
        if t_src is None:
            t_src = np.arange(len(T_src)) / (probe.fps if probe.fps > 0 else hz)
            notes.append("ViPE pose npz had no timestamps; assumed source-fps grid")

        q_src, p_src = se3_to_quat_pos(T_src)
        p, _, valid_p = resample_linear(t_src, p_src, t_dst, max_gap)
        q, _, valid_q = resample_quat_wxyz(t_src, q_src, t_dst, max_gap)
        valid = valid_p & valid_q
        T = se3_from_quat_pos(q, p)
        # ViPE exposes no per-frame confidence at the CLI: constant heuristic.
        conf = np.where(valid, 0.8, 0.1)
        notes.append("conf=0.8 heuristic constant (ViPE has no per-frame confidence)")
        tracked_ratio = float(valid.mean())
        df = sio.poses_df(t_dst, np.arange(n, dtype=np.int64), T, conf, valid, "estimated")
        sio.write_table(df, ws.poses_parquet, required_columns=sio.POSES_COLUMNS)

        # intrinsics ---------------------------------------------------------
        intr_file = self._find_file(base, ("intrinsic",), ".npz")
        if intr_file is None:
            raise RuntimeError(f"no intrinsics npz under {base}; verify ViPE layout")
        intr = sio.read_npz(intr_file)
        if "K" in intr:
            K = np.asarray(intr["K"], np.float64).reshape(3, 3)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        elif all(k in intr for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
        else:
            raise RuntimeError(f"{intr_file} has neither K nor fx/fy/cx/cy")

        # depth --------------------------------------------------------------
        depth_dir = base / "depth"
        depth_files = sorted(depth_dir.glob("*")) if depth_dir.is_dir() else []
        if not depth_files:
            raise RuntimeError(f"no depth outputs under {depth_dir}; verify ViPE layout")
        first = self._load_depth_any(depth_files[0])
        d_h, d_w = first.shape
        cam = CameraInfo(
            model="pinhole", width=probe.width, height=probe.height,
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            depth_scale=1000.0, depth_width=d_w, depth_height=d_h,
            scale_source="vipe_near_metric",
        )
        sio.write_json_model(ws.camera_json, cam)

        # nearest-neighbor map source depth frames onto the canonical grid
        src_idx = np.minimum(
            np.round(t_dst * (probe.fps if probe.fps > 0 else hz)).astype(int),
            len(depth_files) - 1,
        )
        ws.depth_dir.mkdir(parents=True, exist_ok=True)
        coverages = []
        cache_idx, cache = -1, first
        for k in range(n):
            if src_idx[k] != cache_idx:
                cache = self._load_depth_any(depth_files[src_idx[k]])
                cache_idx = src_idx[k]
            sio.write_depth_png(ws.depth_frame(k), cache, cam.depth_scale)
            if k % max(1, n // 10) == 0:
                coverages.append(float((np.nan_to_num(cache) > 0).mean()))
        depth_coverage = float(np.median(coverages)) if coverages else 0.0

        # scene reconstruction ------------------------------------------------
        rng = rng_for(ws.episode_id, self.name)  # only for point subsampling
        kf = np.unique(np.linspace(0, n - 1, 6).astype(int))
        pts = []
        for k in kf:
            d = self._load_depth_any(depth_files[src_idx[k]])
            pts.append(_backproject_world(np.nan_to_num(d), cam, T[k], 20000 // len(kf), rng))
        cloud_pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        trimesh.PointCloud(cloud_pts).export(str(ws.scene_ply))
        try:
            import open3d  # noqa: F401  (TSDF fusion lives in orchestrator env)

            notes.append("open3d present: TSDF fusion TODO at pinned commit")
            _write_scene_mesh(ws.scene_mesh_glb)  # placeholder until TSDF wired
        except ImportError:
            notes.append("open3d unavailable: TSDF scene_mesh skipped, coarse box written")
            _write_scene_mesh(ws.scene_mesh_glb)

        return self._gate_and_result(
            ctx,
            metrics={
                "tracked_ratio": tracked_ratio,
                "depth_coverage": depth_coverage,
                "n_frames": n,
                "notes": "; ".join(notes),
            },
            tool="vipe",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _find_file(base: Path, stems: tuple[str, ...], suffix: str) -> Optional[Path]:
        for p in sorted(base.rglob(f"*{suffix}")):
            if any(s in p.stem.lower() for s in stems):
                return p
        return None

    @staticmethod
    def _load_depth_any(path: Path) -> np.ndarray:
        """Load one raw depth frame (meters) from npz or 16-bit png (mm)."""
        if path.suffix == ".npz":
            z = sio.read_npz(path)
            return np.asarray(next(iter(z.values())), np.float64)
        if path.suffix == ".png":
            return sio.read_depth_png(path, 1000.0)
        raise RuntimeError(f"unsupported ViPE depth format: {path.name}")

    # ------------------------------------------------------------------
    def _gate_and_result(self, ctx: StageContext, metrics: dict, tool: str) -> StageResult:
        ws = ctx.ws
        qa = ctx.cfg.qa.get("geometry", {})
        gate = gate_from_thresholds(metrics, [
            ("tracked_ratio", "ge", float(qa.get("min_tracked_ratio", 0.90)), True),
            ("depth_coverage", "ge", float(qa.get("min_depth_coverage", 0.60)), True),
        ])
        outputs = [
            ws.rel(ws.camera_json),
            ws.rel(ws.poses_parquet),
            ws.rel(ws.depth_dir),
            ws.rel(ws.scene_ply),
            ws.rel(ws.scene_mesh_glb),
        ]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(
            status=status,
            metrics=metrics,
            failure_reason=None if gate.passed else "; ".join(gate.reasons),
            outputs=outputs,
            gate=gate,
            tool=tool,
            repo=VIPE_REPO,
            commit=VIPE_COMMIT if tool == "vipe" else "",
        )
