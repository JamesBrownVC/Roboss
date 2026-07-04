"""`v2r syngen` CLI: request / status / deliver / run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import V2RConfig
from .spec import JobDirs
from .verify import VerificationRecord

syngen_app = typer.Typer(
    name="syngen",
    help="Synthetic data generation loop: prompt -> Gemini/Veo -> verify -> V2R -> LeRobot",
)
console = Console()


def _cfg(root: Optional[Path]) -> V2RConfig:
    return V2RConfig.load(root)


@syngen_app.command("request")
def request_cmd(
    prompt: str = typer.Argument(..., help="Desired training data, e.g. 'person picking up objects from a table'"),
    variants: int = typer.Option(4, "--variants", help="Motion/lighting event variations"),
    cameras: int = typer.Option(2, "--cameras", help="Camera viewpoints per event (multi-view tier)"),
    job_id: Optional[str] = typer.Option(None, "--job-id"),
    backend: str = typer.Option("auto", "--backend", help="mock | omni | veo | auto"),
    duration: int = typer.Option(4, "--duration", help="Seconds per video (veo/mock; omni is model-controlled)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Gemini director; deterministic expansion"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Step 1+2: expand the request into a parametrized spec (spec.json)."""
    from .pipeline import create_request

    cfg = _cfg(root)
    spec, dirs = create_request(
        cfg, prompt, n_variants=variants, n_cameras=cameras, job_id=job_id,
        backend=backend, duration_s=duration, use_llm=not no_llm,
        log=console.print)
    table = Table(title=f"Job {spec.job_id} — {len(spec.variants)} videos planned")
    table.add_column("variant")
    table.add_column("prompt", max_width=90)
    for v in spec.variants:
        table.add_row(v.variant_id, v.prompt)
    console.print(table)
    console.print(f"[green]Spec:[/green] {dirs.spec_json}")
    console.print(f"Next: v2r syngen run --job-id {spec.job_id}")


@syngen_app.command("status")
def status_cmd(
    job_id: str = typer.Option(..., "--job-id"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Show job phase, generation and verification progress."""
    from .pipeline import load_status

    cfg = _cfg(root)
    dirs = JobDirs(cfg.root, job_id)
    if not dirs.root.is_dir():
        console.print(f"[red]No such job:[/red] {dirs.root}")
        raise typer.Exit(code=1)
    status = load_status(dirs)
    console.print_json(json.dumps(status))

    if dirs.spec_json.is_file():
        spec = dirs.load_spec()
        table = Table(title="Variants")
        table.add_column("variant")
        table.add_column("video")
        table.add_column("verdict")
        table.add_column("reasons", max_width=60)
        for v in spec.variants:
            has_video = "yes" if dirs.video_mp4(v.variant_id).is_file() else "-"
            verdict, reasons = "-", ""
            vpath = dirs.verification_json(v.variant_id)
            if vpath.is_file():
                rec = VerificationRecord.model_validate_json(vpath.read_text(encoding="utf-8"))
                verdict = rec.verdict
                reasons = "; ".join(rec.verdict_reasons)
            table.add_row(v.variant_id, has_video, verdict, reasons)
        console.print(table)


@syngen_app.command("deliver")
def deliver_cmd(
    job_id: str = typer.Option(..., "--job-id"),
    robots: str = typer.Option("g1", "--robots"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Steps 3-6 for an existing job: generate/verify (if pending), ingest, deliver."""
    from .pipeline import run_job

    cfg = _cfg(root)
    dirs = JobDirs(cfg.root, job_id)
    if not dirs.spec_json.is_file():
        console.print(f"[red]No spec for job {job_id!r}[/red] — run `v2r syngen request` first")
        raise typer.Exit(code=1)
    robot_list = [r.strip() for r in robots.split(",") if r.strip()]
    delivery = run_job(cfg, job_id=job_id, robots=robot_list, log=console.print)
    console.print(f"[green]Delivery:[/green] {delivery}")


@syngen_app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Desired training data"),
    variants: int = typer.Option(4, "--variants"),
    cameras: int = typer.Option(2, "--cameras"),
    job_id: Optional[str] = typer.Option(None, "--job-id"),
    backend: str = typer.Option("auto", "--backend", help="mock | omni | veo | auto"),
    duration: int = typer.Option(4, "--duration"),
    robots: str = typer.Option("g1", "--robots"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    workers: int = typer.Option(4, "--workers"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """One-shot: request -> generate -> verify -> ingest -> deliver."""
    from .pipeline import run_job

    cfg = _cfg(root)
    robot_list = [r.strip() for r in robots.split(",") if r.strip()]
    delivery = run_job(
        cfg, user_prompt=prompt, job_id=job_id, n_variants=variants,
        n_cameras=cameras, backend=backend, duration_s=duration,
        robots=robot_list, use_llm=not no_llm, max_workers=workers,
        log=console.print)
    console.print(f"\n[green]Delivery folder:[/green] {delivery}")
    card = delivery / "README.md"
    if card.is_file():
        console.print(f"[green]Dataset card:[/green] {card}")
