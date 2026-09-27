from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVICE_ROOT.parents[1] if len(SERVICE_ROOT.parents) > 1 else SERVICE_ROOT


def _path_from_env(name: str, candidates: tuple[Path, ...]) -> Path | None:
    configured = os.getenv(name)
    if configured:
        return Path(configured).expanduser().resolve()
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _directory_from_env(name: str, candidates: tuple[Path, ...]) -> Path | None:
    configured = os.getenv(name)
    if configured:
        return Path(configured).expanduser().resolve()
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


@dataclass(frozen=True, slots=True)
class Settings:
    catalog_path: Path | None
    index_manifest_path: Path | None
    local_feature_index_path: Path | None = None
    neural_index_dir: Path | None = None
    visual_index_dir: Path | None = None
    ocr_model_dir: Path | None = None
    fusion_path: Path | None = None
    images_dir: Path | None = None
    found_probability: float = 0.80
    not_found_probability: float = 0.35
    ocr_enabled: bool = False
    ocr_tessdata_dir: Path | None = None
    ocr_aliases_path: Path | None = None
    ocr_min_score: float = 7.0
    ocr_workers: int = 6
    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 30_000_000
    min_image_side: int = 32
    cache_size: int = 256
    found_threshold: float = 0.72
    model_version: str = "cpu-visual-baseline-v1"
    fallback_slug: str | None = None
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        catalog = _path_from_env(
            "CIFR_CATALOG_PATH",
            (
                SERVICE_ROOT / "data" / "catalog.jsonl",
                SERVICE_ROOT / "data" / "catalog.json",
                REPOSITORY_ROOT / "data" / "generated" / "expanded" / "catalog.jsonl",
                REPOSITORY_ROOT / "data" / "generated" / "catalog" / "catalog.jsonl",
                REPOSITORY_ROOT / "data" / "manifests" / "catalog.jsonl",
                REPOSITORY_ROOT / "Датасет" / "strapi_output0709.csv",
            ),
        )
        index_manifest = _path_from_env(
            "CIFR_INDEX_MANIFEST",
            (
                SERVICE_ROOT / "data" / "index_manifest.json",
                REPOSITORY_ROOT / "data" / "generated" / "expanded" / "index_manifest.json",
                REPOSITORY_ROOT / "data" / "generated" / "index" / "index_manifest.json",
                REPOSITORY_ROOT / "data" / "manifests" / "index_manifest.json",
            ),
        )
        tessdata = _directory_from_env(
            "CIFR_OCR_TESSDATA_DIR",
            (SERVICE_ROOT / "data" / "tessdata",),
        )
        aliases = _path_from_env(
            "CIFR_OCR_ALIASES_PATH",
            (REPOSITORY_ROOT / "data" / "catalog" / "label_aliases.json",),
        )
        return cls(
            catalog_path=catalog,
            local_feature_index_path=None if os.getenv('CIFR_LOCAL_FEATURES_ENABLED', '1') == '0' else _path_from_env(
                'CIFR_LOCAL_FEATURE_INDEX',
                (SERVICE_ROOT / 'data' / 'local_features.npz',
                 REPOSITORY_ROOT / 'data' / 'generated' / 'expanded' / 'local_features.npz'),
            ),
            neural_index_dir=None if os.getenv("CIFR_NEURAL_ENABLED", "1").casefold() in {"0", "false", "no"} else _directory_from_env(
                "CIFR_NEURAL_INDEX_DIR",
                (SERVICE_ROOT / "data" / "dinov2",
                 REPOSITORY_ROOT / "data" / "generated" / "expanded" / "dinov2"),
            ),
            index_manifest_path=index_manifest,
            visual_index_dir=None if os.getenv("CIFR_VISUAL_ENABLED", "1").casefold() in {"0", "false", "no"} else _directory_from_env(
                "CIFR_VISUAL_INDEX_DIR",
                (SERVICE_ROOT / "data" / "siglip",
                 REPOSITORY_ROOT / "data" / "generated" / "expanded" / "siglip"),
            ),
            ocr_model_dir=None if os.getenv("CIFR_OCR_ENABLED", "1").casefold() in {"0", "false", "no"} else _directory_from_env(
                "CIFR_OCR_MODEL_DIR", (SERVICE_ROOT / "data" / "ocr",),
            ),
            fusion_path=_path_from_env(
                "CIFR_FUSION_PATH",
                (SERVICE_ROOT / "data" / "fusion.json",
                 REPOSITORY_ROOT / "data" / "generated" / "expanded" / "fusion.json"),
            ),
            images_dir=_directory_from_env(
                "CIFR_IMAGES_DIR",
                (SERVICE_ROOT / "data" / "images",
                 REPOSITORY_ROOT / "data" / "generated" / "expanded" / "images"),
            ),
            found_probability=float(os.getenv("CIFR_FOUND_PROBABILITY", 0.80)),
            not_found_probability=float(os.getenv("CIFR_NOT_FOUND_PROBABILITY", 0.35)),
            ocr_enabled=os.getenv("CIFR_OCR_ENABLED", "1").casefold() not in {"0", "false", "no"}
            and tessdata is not None,
            ocr_tessdata_dir=tessdata,
            ocr_aliases_path=aliases,
            ocr_min_score=float(os.getenv("CIFR_OCR_MIN_SCORE", 7.0)),
            ocr_workers=int(os.getenv("CIFR_OCR_WORKERS", 6)),
            max_upload_bytes=int(os.getenv("CIFR_MAX_UPLOAD_BYTES", 15 * 1024 * 1024)),
            max_image_pixels=int(os.getenv("CIFR_MAX_IMAGE_PIXELS", 30_000_000)),
            min_image_side=int(os.getenv("CIFR_MIN_IMAGE_SIDE", 32)),
            cache_size=int(os.getenv("CIFR_CACHE_SIZE", 256)),
            found_threshold=float(os.getenv("CIFR_FOUND_THRESHOLD", 0.72)),
            model_version=os.getenv("CIFR_MODEL_VERSION", "cpu-visual-baseline-v1"),
            fallback_slug=os.getenv("CIFR_FALLBACK_SLUG") or None,
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "CIFR_CORS_ORIGINS",
                    "http://localhost:3000,http://localhost:3001,http://localhost:3002,"
                    "http://localhost:3003,http://localhost:5173,http://127.0.0.1:3000,"
                    "http://127.0.0.1:3001,http://127.0.0.1:3002,"
                    "http://127.0.0.1:3003,http://localhost:3004,"
                    "http://127.0.0.1:3004,http://127.0.0.1:5173",
                ).split(",")
                if origin.strip()
            ),
        )
