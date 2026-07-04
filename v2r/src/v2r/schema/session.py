"""Multi-view session workspace layout (master prompt section 7)."""

from __future__ import annotations

from pathlib import Path


class SessionWorkspace:
    """Same-event multi-camera session under workspaces/sessions/{session_id}/."""

    def __init__(self, sessions_root: Path | str, session_id: str):
        self.root = Path(sessions_root) / session_id
        self.session_id = session_id

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    def cam_video(self, cam_id: str) -> Path:
        return self.raw_dir / "cams" / cam_id / "video.mp4"

    def cam_dir(self, cam_id: str) -> Path:
        return self.raw_dir / "cams" / cam_id

    @property
    def sync_json(self) -> Path:
        return self.root / "sync.json"

    @property
    def calibration_json(self) -> Path:
        return self.root / "calibration.json"

    @property
    def triangulated_dir(self) -> Path:
        return self.root / "triangulated"

    @property
    def joints_parquet(self) -> Path:
        return self.triangulated_dir / "joints.parquet"

    @property
    def fused_dir(self) -> Path:
        return self.root / "fused"

    @property
    def monocular_shadow_dir(self) -> Path:
        return self.root / "monocular_shadow"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def cross_view_reproj_json(self) -> Path:
        return self.qa_dir / "cross_view_reproj.json"

    @property
    def session_meta_json(self) -> Path:
        return self.root / "session.json"

    _ALL_DIRS = (
        "raw/cams",
        "triangulated",
        "fused/human",
        "fused/geometry",
        "fused/objects",
        "monocular_shadow",
        "qa",
    )

    def create(self) -> "SessionWorkspace":
        for d in self._ALL_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        return Path(path).relative_to(self.root).as_posix()

    def list_cameras(self) -> list[str]:
        cams_dir = self.raw_dir / "cams"
        if not cams_dir.is_dir():
            return []
        return sorted(
            d.name for d in cams_dir.iterdir()
            if d.is_dir() and (d / "video.mp4").is_file()
        )
