"""Contact (Stage F): signed distance hand fingertips <-> object mesh.

REAL geometric inference (spec 6.F) — the same code runs on synthetic-mode
puppet kinematics and on real estimated kinematics. contact=true if the min
fingertip distance < qa.contact.dist_on_m sustained >= sustain_frames, with
hysteresis at dist_off_m. Penetration depth is recorded as a QA signal.
All fields source=estimated; no forces are fabricated (none exist in RGB).
"""

from __future__ import annotations

from ..labeling.kinematics import infer_contacts
from ..schema.io import read_table, write_table
from ..schema.models import StageStatus
from .base import Stage, StageContext, StageResult, register_stage


@register_stage
class ContactStage(Stage):
    name = "contact"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        if not ws.hands_parquet.is_file() or not ws.tracks_parquet.is_file():
            return StageResult(
                status=StageStatus.failed,
                failure_reason="missing hands.parquet or tracks.parquet",
                tool="geometric_contact", repo="v2r-internal", commit="0.2.0",
            )
        hands = read_table(ws.hands_parquet)
        tracks = read_table(ws.tracks_parquet)

        import trimesh

        meshes = {}
        for oid in tracks["object_id"].astype(str).unique():
            glb = ws.object_mesh_glb(oid)
            if glb.is_file():
                try:
                    loaded = trimesh.load(glb, force="mesh")
                    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
                        meshes[oid] = loaded
                except Exception:
                    pass  # fall back to sphere proxy inside infer_contacts

        qa = ctx.cfg.qa.get("contact", {})
        df = infer_contacts(
            hands, tracks, meshes,
            dist_on_m=float(qa.get("dist_on_m", 0.005)),
            dist_off_m=float(qa.get("dist_off_m", 0.010)),
            sustain_frames=int(qa.get("sustain_frames", 3)),
        )
        write_table(df, ws.contacts_parquet)

        contact_frames = int(df["contact"].sum()) if not df.empty else 0
        flags = df.sort_values(["hand", "object_id", "frame"])["contact"].to_numpy() if not df.empty else []
        n_events = 0
        if len(flags):
            import numpy as np

            keyed = df.sort_values(["hand", "object_id", "frame"])
            for _, grp in keyed.groupby(["hand", "object_id"]):
                f = grp["contact"].to_numpy(dtype=bool)
                n_events += int((f[1:] & ~f[:-1]).sum() + (1 if f[0] else 0))
        metrics = {
            "n_contact_events": n_events,
            "contact_frames": contact_frames,
            "max_penetration_m": float(df["penetration_m"].max()) if not df.empty else 0.0,
            "meshes_loaded": sorted(meshes),
        }
        return StageResult(
            status=StageStatus.success,
            metrics=metrics,
            outputs=[ws.rel(ws.contacts_parquet)],
            tool="geometric_contact", repo="v2r-internal", commit="0.2.0",
        )
