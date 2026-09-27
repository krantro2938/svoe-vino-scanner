from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .imaging import ImageInputError, decode_image
from .models import (
    AnalogWine,
    EvalPrediction,
    RankedWine,
    HealthResponse,
    PairingRequest,
    PairingResponse,
    SearchResponse,
    Wine,
)
from .retrieval import Match, RecognitionEngine
from .sommelier import recommend


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


async def _read_upload(upload: UploadFile, maximum: int) -> bytes:
    payload = await upload.read(maximum + 1)
    await upload.close()
    if len(payload) > maximum:
        raise ImageInputError(
            "payload_too_large", f"The upload exceeds the {maximum:,}-byte limit.", 413
        )
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.startup_error = None
        try:
            application.state.engine = RecognitionEngine(resolved_settings)
            application.state.ready = True
        except Exception as error:
            application.state.startup_error = str(error)
        yield

    application = FastAPI(
        title="CIFR Wine Recognition API",
        version="0.1.0",
        description="Offline, CPU-first image retrieval and grounded wine pairing service.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        # Local demo: the UI may be opened from a phone on the same LAN.
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+"
        r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.middleware("http")
    async def timing_header(request: Request, call_next: Any):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - started) * 1000))
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError):
        missing_image = any(
            item.get("type") == "missing" and "image" in item.get("loc", ())
            for item in error.errors()
        )
        if missing_image:
            return _error(422, "image_required", "Multipart field 'image' is required.")
        return _error(422, "invalid_request", "The request fields are invalid.")

    @application.exception_handler(ImageInputError)
    async def image_error(_: Request, error: ImageInputError):
        return _error(error.status_code, error.code, str(error))

    def engine(request: Request) -> RecognitionEngine:
        if not request.app.state.ready:
            raise RuntimeError(request.app.state.startup_error or "recognizer is not ready")
        return request.app.state.engine

    async def run_prediction(upload: UploadFile, request: Request) -> Match:
        recognizer = engine(request)
        payload = await _read_upload(upload, resolved_settings.max_upload_bytes)
        decoded = decode_image(
            payload,
            max_pixels=resolved_settings.max_image_pixels,
            min_side=resolved_settings.min_image_side,
        )
        digest = recognizer.sha256(payload)
        return recognizer.predict(decoded, digest)

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(request: Request):
        if not request.app.state.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "error": request.app.state.startup_error or "initializing",
                },
            )
        recognizer: RecognitionEngine = request.app.state.engine
        return HealthResponse(
            status="ready",
            catalog_count=len(recognizer.catalog),
            indexed_count=recognizer.indexed_count,
            model_version=recognizer.model_version,
        )

    @application.post(
        "/v1/eval/predict", response_model=EvalPrediction, tags=["recognition"]
    )
    async def eval_predict(request: Request, image: UploadFile) -> EvalPrediction:
        match = await run_prediction(image, request)
        return EvalPrediction(slug=match.wine.slug)

    def card(recognizer: RecognitionEngine, wine: Wine) -> Wine:
        """Attach the API URL of the catalogue reference photo, when present."""
        if wine.photo_name and recognizer.image_path(wine.slug) is not None:
            return wine.model_copy(update={"image_url": f"/v1/wines/{wine.slug}/image"})
        return wine

    def status_for(match: Match, recognizer: RecognitionEngine) -> str:
        if not match.candidates:   # legacy cascade: heuristic confidence
            if not match.indexed:
                return "uncertain"
            if match.quality_hint and match.confidence < resolved_settings.found_threshold:
                return "low_quality"
            return "found" if match.confidence >= resolved_settings.found_threshold else "uncertain"
        if match.confidence >= resolved_settings.found_probability:
            return "found"
        if match.quality_hint:
            return "low_quality"
        if match.confidence < recognizer.not_found_probability:
            return "not_found"
        return "uncertain"

    @application.post("/v1/search", response_model=SearchResponse, tags=["recognition"])
    async def search(request: Request, image: UploadFile) -> SearchResponse:
        started = time.perf_counter()
        match = await run_prediction(image, request)
        elapsed = round((time.perf_counter() - started) * 1000)
        recognizer = engine(request)
        status = status_for(match, recognizer)
        top5 = [
            RankedWine(rank=i + 1, probability=probability, wine=card(recognizer, recognizer.catalog[slug]))
            for i, (slug, probability) in enumerate(match.candidates)
        ]
        analogs = []
        if status != "found":
            analogs = [
                AnalogWine(wine=card(recognizer, wine), reasons=reasons)
                for wine, reasons in recognizer.analogs.similar(match.wine.slug, limit=4, other_wineries=False)
            ]
        return SearchResponse(
            status=status,
            wine=card(recognizer, match.wine),
            confidence=match.confidence,
            confidence_top5=round(min(1.0, sum(p for _, p in match.candidates)), 4),
            top1_top2_margin=match.margin,
            top5=top5,
            analogs=analogs,
            latency_ms=elapsed,
            stage_timings_ms=match.timings_ms or {},
            model_version=recognizer.model_version,
            method=match.method,
            quality_hint=match.quality_hint,
        )

    @application.get("/v1/wines/{slug}/image", tags=["catalog"])
    async def wine_image(slug: str, request: Request):
        path = engine(request).image_path(slug)
        if path is None:
            return _error(404, "image_not_found", "No reference photo for that slug.")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @application.get("/v1/wines/{slug}/analogs", response_model=list[AnalogWine], tags=["catalog"])
    async def wine_analogs(slug: str, request: Request, limit: int = 4):
        recognizer = engine(request)
        if slug not in recognizer.catalog:
            return _error(404, "wine_not_found", "No catalog wine has that slug.")
        return [
            AnalogWine(wine=card(recognizer, wine), reasons=reasons)
            for wine, reasons in recognizer.analogs.similar(slug, limit=max(1, min(limit, 12)))
        ]

    @application.get("/v1/wines/{slug}", response_model=Wine, tags=["catalog"])
    async def get_wine(slug: str, request: Request):
        recognizer = engine(request)
        wine = recognizer.catalog.get(slug)
        if wine is None:
            return _error(404, "wine_not_found", "No catalog wine has that slug.")
        return card(recognizer, wine)

    @application.post(
        "/v1/sommelier/pairing", response_model=PairingResponse, tags=["sommelier"]
    )
    async def pairing(payload: PairingRequest, request: Request):
        recognizer = engine(request)
        if payload.slug not in recognizer.catalog:
            return _error(404, "wine_not_found", "No catalog wine has that slug.")
        return recommend(payload, recognizer.catalog, recognizer.analogs)

    return application


app = create_app()
