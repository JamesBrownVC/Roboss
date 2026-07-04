"""V2R CLI: pipeline run, import videos, extract timeseries, multi-view sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import V2RConfig
from ..import_.downloader import import_all
from ..orchestrator.runner import resolve_stages, run_episode
from ..session.runner import (
    parse_cam_spec,
    run_session,
    session_calibrate,
    session_create,
    session_fuse,
    session_sync,
    session_triangulate,
)
from ..syngen.cli import syngen_app
from ..timeseries.export_training import build_training_npz
from ..timeseries.extract import extract_all

app = typer.Typer(name="v2r", help="Video -> robot-ready (LeRobot v3) pipeline")
session_app = typer.Typer(name="session", help="Multi-view same-event sessions (GT tier)")
app.add_typer(session_app, name="session")
app.add_typer(syngen_app, name="syngen")
console = Console()


def _repo_root(root: Optional[Path]) -> Path:
    if root is not None:
        return Path(root).resolve()
    return V2RConfig.load().root


@app.command("label")
def label_video(
    video: Path = typer.Option(..., "--video", help="Path to any video (AI-generated or real)"),
    episode_id: Optional[str] = typer.Option(None, "--episode-id", help="Workspace episode id (default: derived from filename)"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model override"),
    root: Optional[Path] = typer.Option(None, "--root", help="V2R repo root"),
):
    """Agentic labeling: VLM plans which perception tools to run (MediaPipe
    pose/hands, YOLO, motion analysis), runs them, then writes segments,
    captions, scene tags and a feasibility verdict into the workspace."""
    from ..agentic import run_agentic_labeler

    cfg = V2RConfig.load(root)
    report = run_agentic_labeler(cfg, Path(video), episode_id=episode_id,
                                 model=model, log=console.print)
    feas = report["feasibility"]
    table = Table(title=f"Agentic labels - {report['episode_id']}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("judge", report["judge_source"])
    table.add_row("human present", str(feas.get("human_present")))
    table.add_row("AI-generated suspected", str(feas.get("ai_generated_suspected")))
    table.add_row("recommendation", str(feas.get("recommendation")))
    table.add_row("confidence", f"{feas.get('confidence', 0):.2f}")
    console.print(table)


@app.command("run")
def run_pipeline(
    episode: Path = typer.Option(..., "--episode", help="Path to source video (or glob — single file for now)"),
    stages: str = typer.Option("all", "--stages", help="all or comma-separated stage names"),
    robots: str = typer.Option("g1", "--robots", help="Comma-separated robot names from config/robots.yaml"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Override mode: synthetic | real"),
    root: Optional[Path] = typer.Option(None, "--root", help="V2R repo root"),
):
    """Run the V2R research pipeline on one episode (synthetic mode works on any host)."""
    cfg = V2RConfig.load(root)
    robot_list = [r.strip() for r in robots.split(",") if r.strip()]
    stage_set = resolve_stages(stages)
    console.print(f"[bold]V2R run[/bold] episode={episode} robots={robot_list} stages={stages}")
    result = run_episode(
        cfg,
        Path(episode),
        robots=robot_list,
        stages=stage_set,
        mode_override=mode,
        log=console.print,
    )
    table = Table(title=f"Episode {result.episode_id}")
    table.add_column("Stage")
    table.add_column("Status")
    for name, status in result.stages.items():
        table.add_row(name, status.value)
    console.print(table)
    console.print(f"\nWorkspace: {result.workspace}")
    color = "green" if result.accepted else "yellow"
    console.print(f"[{color}]Accepted:[/{color}] {result.accepted}")
    if result.errors:
        console.print(f"[red]Errors:[/red] {result.errors}")
    raise typer.Exit(code=0 if result.accepted else 1)


@app.command("import-datasets")
def import_datasets(
    subject: Optional[str] = typer.Option(None, help="human | animal"),
    source: Optional[list[str]] = typer.Option(None, "--source", "-s", help="Source id(s) from datasets.yaml"),
    root: Optional[Path] = typer.Option(None, "--root", help="V2R repo root"),
):
    """Download human and animal videos from the curated catalog."""
    repo = _repo_root(root)
    console.print(f"[bold]Importing videos[/bold] -> {repo / 'data' / 'raw'}")
    results, data_root = import_all(repo, subject=subject, source_ids=source, log=console.print)
    table = Table(title="Import results")
    table.add_column("Source")
    table.add_column("Subject")
    table.add_column("Videos", justify="right")
    table.add_column("Errors", justify="right")
    for r in results:
        table.add_row(r.source_id, r.subject, str(len(r.videos)), str(len(r.errors)))
    console.print(table)
    total = sum(len(r.videos) for r in results)
    console.print(f"\n[green]Done:[/green] {total} videos in {data_root}")


@app.command("extract-timeseries")
def extract_timeseries_cmd(
    subject: Optional[str] = typer.Option(None, help="human | animal"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Extract pose/motion timeseries from imported videos (MediaPipe + YOLO)."""
    repo = _repo_root(root)
    console.print("[bold]Extracting timeseries[/bold] (human=MediaPipe pose, animal=YOLO track)")
    results, out_root = extract_all(repo, subject=subject, log=console.print)
    ok = sum(1 for r in results if not r.errors)
    console.print(f"\n[green]Done:[/green] {ok}/{len(results)} episodes -> {out_root}")


@app.command("build-training-set")
def build_training_set(
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Pack parquet timeseries into NPZ arrays ready for model training."""
    repo = _repo_root(root)
    console.print("[bold]Building training NPZ bundles[/bold]")
    out = build_training_npz(repo, log=console.print)
    console.print(f"\n[green]Done:[/green] {out}")


@app.command("pipeline")
def full_pipeline(
    subject: Optional[str] = typer.Option(None, help="human | animal | all"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Import -> extract timeseries -> build training NPZ (full workflow)."""
    subj = None if subject in (None, "all") else subject
    import_datasets(subject=subj, source=None, root=root)
    extract_timeseries_cmd(subject=subj, root=root)
    build_training_set(root=root)


def _session_ws(cfg: V2RConfig, session_id: str):
    from ..schema.session import SessionWorkspace
    return SessionWorkspace(cfg.workspaces_root / "sessions", session_id)


@session_app.command("create")
def session_create_cmd(
    id: str = typer.Option(..., "--id", help="Session identifier"),
    videos: list[str] = typer.Option(..., "--videos", help="cam0:path.mp4 cam1:path.mp4 ..."),
    variants: Optional[list[str]] = typer.Option(None, "--variants", help="cam0:gen1.mp4 cam0:gen2.mp4 ..."),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Create a multi-view session workspace with per-camera videos."""
    cfg = V2RConfig.load(root)
    cam_videos = parse_cam_spec(videos)
    var_map: dict[str, list[Path]] = {}
    if variants:
        for spec in variants:
            cam_id, path = spec.split(":", 1)
            var_map.setdefault(cam_id.strip(), []).append(Path(path.strip()).resolve())
    sw = session_create(cfg, id, cam_videos, variants=var_map or None, log=console.print)
    console.print(f"[green]Session created:[/green] {sw.root}")


@session_app.command("sync")
def session_sync_cmd(
    id: str = typer.Option(..., "--id"),
    mode: Optional[str] = typer.Option(None, "--mode"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Synchronize cameras via audio cross-correlation (+ timecode when available)."""
    cfg = V2RConfig.load(root)
    sw = _session_ws(cfg, id)
    sync = session_sync(sw, cfg, mode=mode or cfg.pipeline.default_mode)
    console.print(f"sync method={sync.method} confidence={sync.confidence:.2f}")
    for cam in sync.cameras:
        console.print(f"  {cam.cam_id}: offset_s={cam.offset_s:+.4f} conf={cam.confidence:.2f}")


@session_app.command("calibrate")
def session_calibrate_cmd(
    id: str = typer.Option(..., "--id"),
    mode: Optional[str] = typer.Option(None, "--mode"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Estimate intrinsics/extrinsics (checkerboard, COLMAP stub, or synthetic)."""
    cfg = V2RConfig.load(root)
    sw = _session_ws(cfg, id)
    cal = session_calibrate(sw, cfg, mode=mode or cfg.pipeline.default_mode)
    console.print(f"calibration method={cal.method} cameras={len(cal.cameras)}")


@session_app.command("triangulate")
def session_triangulate_cmd(
    id: str = typer.Option(..., "--id"),
    mode: Optional[str] = typer.Option(None, "--mode"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Triangulate 2D keypoints across views -> joints.parquet (source=triangulated)."""
    cfg = V2RConfig.load(root)
    sw = _session_ws(cfg, id)
    df = session_triangulate(sw, cfg, mode=mode or cfg.pipeline.default_mode)
    console.print(f"[green]Triangulated[/green] {len(df)} joint-rows -> {sw.joints_parquet}")


@session_app.command("fuse")
def session_fuse_cmd(
    id: str = typer.Option(..., "--id"),
    robots: str = typer.Option("g1", "--robots"),
    mode: Optional[str] = typer.Option(None, "--mode"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Fuse triangulated GT into ego view; run monocular shadow benchmark."""
    cfg = V2RConfig.load(root)
    sw = _session_ws(cfg, id)
    robot_list = [r.strip() for r in robots.split(",") if r.strip()]
    eid = session_fuse(sw, cfg, robot_list, mode=mode or cfg.pipeline.default_mode, log=console.print)
    console.print(f"[green]Fused[/green] episode_id={eid}")


@session_app.command("run")
def session_run_cmd(
    id: str = typer.Option(..., "--id"),
    tier: str = typer.Option("multiview", "--tier", help="multiview | multiview_gt"),
    robots: str = typer.Option("g1", "--robots"),
    mode: Optional[str] = typer.Option(None, "--mode"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Run full multi-view DAG: sync -> calibrate -> triangulate -> fuse."""
    cfg = V2RConfig.load(root)
    robot_list = [r.strip() for r in robots.split(",") if r.strip()]
    result = run_session(
        cfg, id, tier=tier, robots=robot_list,
        mode=mode or cfg.pipeline.default_mode,
        log=console.print,
    )
    table = Table(title=f"Session {id}")
    table.add_column("Step")
    table.add_column("Status")
    for name, status in result.steps.items():
        table.add_row(name, status)
    console.print(table)
    if result.errors:
        console.print(f"[red]Errors:[/red] {result.errors}")
        raise typer.Exit(code=1)
    console.print(f"[green]Session workspace:[/green] {result.workspace}")


def main():
    app()


if __name__ == "__main__":
    main()
