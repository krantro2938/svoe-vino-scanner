from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Wine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slug: str
    name: str
    winery: str | None = None
    category: str | None = None
    color: str | None = None
    region: str | None = None
    grapes: str | None = None
    description: str | None = None
    photo_name: str | None = None
    image_url: str | None = None
    # Internal ingestion gate. It is deliberately excluded from public API
    # responses but preserved while building a retrieval index.
    index_ready: bool = Field(default=True, exclude=True)


class EvalPrediction(BaseModel):
    slug: str


class RankedWine(BaseModel):
    rank: int
    probability: float = Field(ge=0.0, le=1.0)
    wine: Wine


class AnalogWine(BaseModel):
    wine: Wine
    reasons: list[str]


class SearchResponse(BaseModel):
    status: Literal["found", "uncertain", "not_found", "low_quality"]
    wine: Wine
    # Fusion-ranker probability of the top-1 card and the probability mass of
    # the top-5 list; the margin is p(top1) - p(top2).
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_top5: float = Field(default=0.0, ge=0.0, le=1.0)
    # Probability that the photographed wine is in the catalogue at all.
    in_catalogue: float | None = Field(default=None, ge=0.0, le=1.0)
    top1_top2_margin: float = Field(ge=0.0, le=1.0)
    top5: list[RankedWine] = Field(default_factory=list)
    analogs: list[AnalogWine] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    model_version: str
    method: str = "fusion"
    quality_hint: str | None = None


class PairingRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=300)
    dish: str = Field(min_length=1, max_length=120)
    sauce: str | None = Field(default=None, max_length=120)
    preference: str | None = Field(default=None, max_length=120)
    occasion: str | None = Field(default=None, max_length=40)


class PairingAlternative(BaseModel):
    slug: str
    name: str
    winery: str | None = None
    category: str | None = None
    reason: str


class PairingResponse(BaseModel):
    slug: str
    verdict: Literal["excellent", "good", "acceptable", "not_ideal"]
    explanation: str
    serving_temperature: str
    glass: str | None = None
    pair_with: list[str]
    alternatives: list[PairingAlternative]
    grounded_in: dict[str, str]
    rules_version: str = "pairing-rules-v1"


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "not_ready"]
    catalog_count: int | None = None
    indexed_count: int | None = None
    model_version: str | None = None
