"""Download the SuperAnimal-Quadruped weights used by the `animal_pose` tool.

The weights (~180 MB total) are gitignored (*.pt); run this once after cloning
to populate ``assets/superanimal/``:

    python scripts/fetch_superanimal.py

Requires ``dlclibrary`` (pip install dlclibrary). No full DeepLabCut install is
needed - inference is done standalone in ``v2r.agentic.superanimal``.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    from dlclibrary import download_huggingface_model

    target = Path(__file__).resolve().parents[1] / "assets" / "superanimal"
    target.mkdir(parents=True, exist_ok=True)
    for model in (
        "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn",
        "superanimal_quadruped_resnet_50",
    ):
        print(f"downloading {model} -> {target}")
        download_huggingface_model(model, str(target))
    print("done. files:")
    for f in sorted(target.glob("*.pt")):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
