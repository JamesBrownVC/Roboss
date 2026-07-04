"""Pydantic models shared by the FastAPI app and service layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompileRequest(BaseModel):
    intention: str
    count: int | None = Field(default=None, ge=1, le=20)
    outdir: str | None = None
    start_frames: bool = True


class VideoPipelineRequest(BaseModel):
    prompt: str
    outdir: str | None = None
    scenario_path: str | None = None
    scenario: dict[str, Any] | None = None
    gate2: bool = True
    label: bool = True
    device: str | None = None
    export_robot_data: bool = False
    robots: list[str] = Field(default_factory=lambda: ["g1"])
    robot_data_mode: str = "synthetic"
    robot_data_stages: str = "all"


class E2ERequest(BaseModel):
    intention: str
    count: int | None = Field(default=3, ge=1, le=20)
    run_name: str | None = None
    start_frames: bool = True
    gate2: bool = True
    label: bool = True
    device: str | None = None
    export_robot_data: bool = False
    robots: list[str] = Field(default_factory=lambda: ["g1"])
    robot_data_mode: str = "synthetic"
    robot_data_stages: str = "all"


class RobotDataExportRequest(BaseModel):
    video_path: str
    outdir: str | None = None
    robots: list[str] = Field(default_factory=lambda: ["g1"])
    mode: str = "synthetic"
    stages: str = "all"


class PipelineResponse(BaseModel):
    status: str
    outdir: str
    data: dict[str, Any]
