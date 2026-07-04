"""Dataset catalog loader for human/animal video sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DirectUrlSource(BaseModel):
    id: str
    subject: Literal["human", "animal"]
    license: str
    description: str = ""
    method: Literal["direct_url"] = "direct_url"
    urls: list[str]


class HuggingFaceSource(BaseModel):
    id: str
    subject: Literal["human", "animal"]
    license: str
    description: str = ""
    method: Literal["huggingface"] = "huggingface"
    dataset: str
    split: str = "train"
    video_column: str = "video"
    max_episodes: int = 10
    note: str = ""


class YouTubeSource(BaseModel):
    id: str
    subject: Literal["human", "animal"]
    license: str
    description: str = ""
    method: Literal["youtube"] = "youtube"
    urls: list[str]


DatasetSource = DirectUrlSource | HuggingFaceSource | YouTubeSource


class ExtractionConfig(BaseModel):
    canonical_hz: float = 30.0
    human: dict[str, Any] = Field(default_factory=dict)
    animal: dict[str, Any] = Field(default_factory=dict)


class DatasetCatalog(BaseModel):
    data_root: str = "data/raw"
    sources: list[dict[str, Any]] = Field(default_factory=list)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    def parsed_sources(self) -> list[DatasetSource]:
        out: list[DatasetSource] = []
        for raw in self.sources:
            method = raw.get("method", "direct_url")
            if method == "direct_url":
                out.append(DirectUrlSource.model_validate(raw))
            elif method == "huggingface":
                out.append(HuggingFaceSource.model_validate(raw))
            elif method == "youtube":
                out.append(YouTubeSource.model_validate(raw))
        return out

    def by_subject(self, subject: Optional[str] = None) -> list[DatasetSource]:
        sources = self.parsed_sources()
        if subject is None:
            return sources
        return [s for s in sources if s.subject == subject]


def load_catalog(root: Path) -> DatasetCatalog:
    path = root / "config" / "datasets.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DatasetCatalog.model_validate(data)
