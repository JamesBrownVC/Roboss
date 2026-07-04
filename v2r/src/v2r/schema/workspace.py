"""Episode workspace layout (master prompt section 4). Single source of truth
for every artifact path; stages never build paths by hand."""

from __future__ import annotations

from pathlib import Path


class EpisodeWorkspace:
    def __init__(self, workspaces_root: Path | str, episode_id: str):
        self.root = Path(workspaces_root) / episode_id
        self.episode_id = episode_id

    @staticmethod
    def make_episode_id(source_id: str, clip_idx: int) -> str:
        return f"{source_id}_{clip_idx:06d}"

    # raw/ -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path: return self.root / "raw"
    @property
    def video_path(self) -> Path: return self.raw_dir / "video.mp4"
    @property
    def probe_path(self) -> Path: return self.raw_dir / "probe.json"
    @property
    def consent_path(self) -> Path: return self.raw_dir / "consent.json"
    def multiview_cam_dir(self, cam_id: str) -> Path:
        return self.raw_dir / "cams" / cam_id

    # geometry/ ----------------------------------------------------------
    @property
    def geometry_dir(self) -> Path: return self.root / "geometry"
    @property
    def camera_json(self) -> Path: return self.geometry_dir / "camera.json"
    @property
    def poses_parquet(self) -> Path: return self.geometry_dir / "poses.parquet"
    @property
    def depth_dir(self) -> Path: return self.geometry_dir / "depth"
    def depth_frame(self, frame: int) -> Path:
        return self.depth_dir / f"{frame:06d}.png"
    @property
    def scene_ply(self) -> Path: return self.geometry_dir / "scene.ply"
    @property
    def scene_mesh_glb(self) -> Path: return self.geometry_dir / "scene_mesh.glb"

    # human/ -------------------------------------------------------------
    @property
    def human_dir(self) -> Path: return self.root / "human"
    @property
    def smplx_npz(self) -> Path: return self.human_dir / "smplx.npz"
    @property
    def hands_parquet(self) -> Path: return self.human_dir / "hands.parquet"
    @property
    def fusion_report_json(self) -> Path: return self.human_dir / "fusion_report.json"

    # objects/ -----------------------------------------------------------
    @property
    def objects_dir(self) -> Path: return self.root / "objects"
    @property
    def tracks_parquet(self) -> Path: return self.objects_dir / "tracks.parquet"
    @property
    def masks_dir(self) -> Path: return self.objects_dir / "masks"
    def object_masks_dir(self, object_id: str) -> Path:
        return self.masks_dir / object_id
    @property
    def object_meshes_dir(self) -> Path: return self.objects_dir / "meshes"
    def object_mesh_glb(self, object_id: str) -> Path:
        return self.object_meshes_dir / f"{object_id}.glb"

    # contact/ -----------------------------------------------------------
    @property
    def contact_dir(self) -> Path: return self.root / "contact"
    @property
    def contacts_parquet(self) -> Path: return self.contact_dir / "contacts.parquet"

    # semantics/ -----------------------------------------------------------
    @property
    def semantics_dir(self) -> Path: return self.root / "semantics"
    @property
    def segments_json(self) -> Path: return self.semantics_dir / "segments.json"
    @property
    def captions_json(self) -> Path: return self.semantics_dir / "captions.json"
    @property
    def scene_tags_json(self) -> Path: return self.semantics_dir / "scene_tags.json"

    # retargets/ -----------------------------------------------------------
    @property
    def retargets_dir(self) -> Path: return self.root / "retargets"
    def retarget_dir(self, robot: str) -> Path:
        return self.retargets_dir / robot
    def qpos_parquet(self, robot: str) -> Path:
        return self.retarget_dir(robot) / "qpos.parquet"
    def qpos_csv(self, robot: str) -> Path:
        # downstream trackers (BeyondMimic) expect CSV (master prompt H1)
        return self.retarget_dir(robot) / "qpos.csv"
    def ee_parquet(self, robot: str) -> Path:
        return self.retarget_dir(robot) / "ee.parquet"
    def mapping_json(self, robot: str) -> Path:
        return self.retarget_dir(robot) / "mapping.json"
    def physics_report_json(self, robot: str) -> Path:
        return self.retarget_dir(robot) / "physics_report.json"

    # qa/ -----------------------------------------------------------------
    @property
    def qa_dir(self) -> Path: return self.root / "qa"
    @property
    def crosschecks_json(self) -> Path: return self.qa_dir / "crosschecks.json"
    @property
    def frames_review_dir(self) -> Path: return self.qa_dir / "frames_review"
    @property
    def decision_json(self) -> Path: return self.qa_dir / "decision.json"
    @property
    def yield_report_md(self) -> Path: return self.qa_dir / "yield_report.md"
    @property
    def feasibility_report_json(self) -> Path: return self.qa_dir / "feasibility_report.json"
    @property
    def feasibility_mask_parquet(self) -> Path: return self.qa_dir / "feasibility_mask.parquet"

    # export/ ---------------------------------------------------------------
    @property
    def export_dir(self) -> Path: return self.root / "export"
    @property
    def lerobot_dir(self) -> Path: return self.export_dir / "lerobot"
    @property
    def egodex_mirror_dir(self) -> Path: return self.export_dir / "egodex_mirror"

    # manifests/ --------------------------------------------------------------
    @property
    def manifests_dir(self) -> Path: return self.root / "manifests"
    def manifest_path(self, stage: str) -> Path:
        return self.manifests_dir / f"{stage}.manifest.json"

    # ------------------------------------------------------------------
    _ALL_DIRS = (
        "raw", "geometry", "geometry/depth", "human",
        "objects", "objects/masks", "objects/meshes",
        "contact", "semantics", "retargets",
        "qa", "qa/frames_review", "export", "manifests",
    )

    def create(self) -> "EpisodeWorkspace":
        for d in self._ALL_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

    def validate_layout(self) -> list[str]:
        """Return list of missing top-level directories (empty == valid)."""
        return [d for d in self._ALL_DIRS if not (self.root / d).is_dir()]

    def rel(self, path: Path) -> str:
        """Workspace-relative POSIX-style path (for manifests)."""
        return Path(path).relative_to(self.root).as_posix()
