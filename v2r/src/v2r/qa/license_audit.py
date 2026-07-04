"""Generate LICENSE_AUDIT.md from config/licensing.yaml."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..config import V2RConfig


def generate_license_audit(repo_root: Path, cfg: V2RConfig) -> Path:
    out = repo_root / "LICENSE_AUDIT.md"
    permissive = cfg.licensing.get("permissive_only", False)
    audit = cfg.licensing.get("audit", [])
    fallbacks = cfg.licensing.get("permissive_fallbacks", {})

    lines = [
        "# V2R License Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**permissive_only**: `{permissive}`",
        "",
        "## Third-party components",
        "",
        "| Name | Kind | Ref | License | Commercial |",
        "|------|------|-----|---------|------------|",
    ]
    for entry in audit:
        lines.append(
            f"| {entry.get('name','')} | {entry.get('kind','')} | {entry.get('ref','')} "
            f"| {entry.get('license','')} | {entry.get('commercial','verify')} |"
        )

    if fallbacks:
        lines.extend(["", "## Permissive-only fallbacks", ""])
        for name, fb in fallbacks.items():
            lines.append(f"- **{name}** → {fb.get('replacement')} (conf×{fb.get('conf_multiplier', 1.0)})")

    lines.extend([
        "",
        "## Operator-provided assets (never scraped)",
        "",
        "- SMPL-X: place registered models in `assets/body_models/`",
        "- MANO: place registered models in `assets/body_models/mano/`",
        "- Robot URDF/MJCF: `assets/robots/{name}/`",
        "",
        "## Notes",
        "",
        "FoundationPose and BundleSDF are NVIDIA non-commercial. "
        "Set `licensing.permissive_only: true` in config for Open3D ICP fallbacks.",
    ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
