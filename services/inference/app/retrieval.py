from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import SCENE_PHOTO_SLUGS, load_catalog
from .config import Settings
from .features import DESCRIPTOR_VERSION, extract_descriptors, image_quality
from .imaging import DecodedImage, center_wine_view
from .models import Wine
from .ocr import LabelOCR, OCRMatch


@dataclass(frozen=True, slots=True)
class Match:
    wine: Wine
    confidence: float
    margin: float
    quality_hint: str | None
    indexed: bool
    method: str = "visual"
    # Ranked (slug, probability) pairs from the fusion ranker, best first.
    candidates: tuple[tuple[str, float], ...] = ()
    timings_ms: dict[str, float] | None = None
    in_catalogue: float | None = None


def _ocr_confidence(match: OCRMatch, visually_indexed: bool) -> float:
    """Convert lexical evidence to a conservative, bounded confidence."""
    score_strength = 1.0 - math.exp(-match.score / 12.0)
    margin_strength = 1.0 - math.exp(-match.margin / 8.0)
    confidence = min(0.96, 0.50 + 0.28 * score_strength + 0.18 * margin_strength)
    if match.producer_conflict or match.vintage_conflict:
        confidence = min(confidence, 0.55)
    elif match.alias_dependent:
        confidence = min(confidence, 0.69)
    elif match.margin < 1.0:
        confidence = min(confidence, 0.75)
    if not visually_indexed:
        confidence = min(confidence, 0.69)
    return confidence


class RecognitionEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.catalog = load_catalog(settings.catalog_path)
        self.model_version = settings.model_version
        self._slugs: list[str] = []
        self._matrix: np.ndarray | None = None
        self._cache: OrderedDict[str, Match] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._load_manifest(settings.index_manifest_path)
        self._indexed_slugs = set(self._slugs)
        self._ocr: LabelOCR | None = None
        if settings.ocr_enabled and settings.ocr_tessdata_dir is not None:
            self._ocr = LabelOCR(
                self.catalog,
                settings.ocr_tessdata_dir,
                settings.ocr_aliases_path,
                settings.ocr_workers,
            )
            self.model_version = "hybrid-ocr-visual-v3-center"

        self._local_features = None
        if settings.local_feature_index_path is not None:
            try:
                from .local_features import LocalFeatures
                local = LocalFeatures()
                local.load(settings.local_feature_index_path, self.catalog)
                self._local_features = local
                self.model_version = "rootsift-hybrid-v1"
            except (ImportError, OSError, ValueError, KeyError) as error:
                logging.getLogger(__name__).warning(
                    "Local feature index unavailable; using OCR/visual fallback: %s", error
                )

        self._neural_features = None
        if settings.neural_index_dir is not None:
            try:
                from .neural_features import NeuralFeatures

                self._neural_features = NeuralFeatures(settings.neural_index_dir, self.catalog)
                self.model_version = "rootsift-dinov2-ocr-v1"
            except (ImportError, OSError, ValueError, KeyError) as error:
                logging.getLogger(__name__).warning(
                    "Neural index unavailable; using local/OCR/visual fallback: %s", error
                )

        self._recognizer = None
        if settings.visual_index_dir is not None:
            try:
                from .pipeline import Recognizer

                self._recognizer = Recognizer(
                    self.catalog,
                    settings.visual_index_dir,
                    settings.ocr_model_dir,
                    self._local_features,
                    settings.fusion_path,
                )
                self._indexed_slugs = set(self._recognizer.visual.slugs)
                self.model_version = "siglip2-rootsift-ppocr-fusion-v1"
                # First inference allocates ONNX/OCR buffers; pay that at startup
                # rather than on the first timed evaluator request.
                from PIL import Image as _Image
                self._recognizer.recognize(_Image.new("RGB", (768, 1024), (128, 96, 80)))
            except (ImportError, OSError, ValueError, KeyError) as error:
                logging.getLogger(__name__).warning(
                    "Fusion recognizer unavailable; using legacy cascade: %s", error
                )

        from .analogs import AnalogFinder

        self.analogs = AnalogFinder(
            self.catalog, self._recognizer.visual if self._recognizer is not None else None
        )

        configured_fallback = settings.fallback_slug
        if configured_fallback and configured_fallback not in self.catalog:
            raise ValueError(f"CIFR_FALLBACK_SLUG is absent from the catalog: {configured_fallback}")
        self.fallback_slug = configured_fallback or sorted(self.catalog)[0]

    @property
    def indexed_count(self) -> int:
        if self._recognizer is not None:
            return len(self._recognizer.visual.slugs)
        return len(set(self._slugs))

    def _load_manifest(self, path: Path | None) -> None:
        if path is None:
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
            raise ValueError("index manifest must be an object containing an items list")
        version = raw.get("descriptor_version")
        if version != DESCRIPTOR_VERSION:
            raise ValueError(
                f"unsupported descriptor version {version!r}; expected {DESCRIPTOR_VERSION!r}"
            )
        if isinstance(raw.get("model_version"), str):
            self.model_version = raw["model_version"]

        vectors: list[np.ndarray] = []
        slugs: list[str] = []
        for item in raw["items"]:
            if not isinstance(item, dict) or item.get("slug") not in self.catalog:
                continue
            descriptor_values = item.get("descriptors")
            if descriptor_values is None and item.get("descriptor") is not None:
                descriptor_values = [item["descriptor"]]
            if not isinstance(descriptor_values, list):
                continue
            for values in descriptor_values:
                vector = np.asarray(values, dtype=np.float32)
                if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                    raise ValueError(f"invalid descriptor for slug {item['slug']}")
                norm = float(np.linalg.norm(vector))
                if not norm:
                    continue
                if vectors and vector.shape != vectors[0].shape:
                    raise ValueError("all index descriptors must have the same dimensions")
                vectors.append(vector / norm)
                slugs.append(item["slug"])
        if vectors:
            self._matrix = np.stack(vectors)
            self._slugs = slugs

    def predict(self, decoded: DecodedImage, payload_sha256: str | None = None) -> Match:
        digest = payload_sha256
        if digest:
            with self._cache_lock:
                cached = self._cache.get(digest)
                if cached is not None:
                    self._cache.move_to_end(digest)
                    return cached

        recognition = None
        if self._recognizer is not None:
            try:
                recognition = self._recognizer.recognize(decoded.image)
            except Exception:  # noqa: BLE001 - an answer beats a 500 for the evaluator
                logging.getLogger(__name__).exception("fusion recognizer failed; using legacy cascade")
        if recognition is not None:
            best = recognition.best
            _, quality_hint = image_quality(decoded.image)
            result = Match(
                self.catalog[best.slug],
                round(best.probability, 4),
                round(recognition.margin, 4),
                quality_hint,
                True,
                "fusion",
                tuple((c.slug, round(c.probability, 4)) for c in recognition.candidates[:5]),
                {k: round(v, 1) for k, v in recognition.timings_ms.items()},
                None if recognition.in_catalogue is None else round(recognition.in_catalogue, 4),
            )
            self._remember(digest, result)
            return result

        if self._local_features is not None:
            from .local_features import PARAMETERS, strong_match
            local_image = decoded.image.copy()
            local_image.thumbnail((PARAMETERS['max_side'], PARAMETERS['max_side']))
            evidence = strong_match(self._local_features.predict(np.asarray(local_image)))
            if evidence is not None:
                slug, confidence, margin = evidence
                _, quality_hint = image_quality(decoded.image)
                result = Match(self.catalog[slug], round(confidence, 4), round(margin, 4),
                               quality_hint, True, "local_features")
                self._remember(digest, result)
                return result

        target_image = center_wine_view(decoded.image)
        quality, quality_hint = image_quality(target_image)
        if self._matrix is None:
            result = Match(self.catalog[self.fallback_slug], 0.0, 0.0, quality_hint, False)
        else:
            queries = extract_descriptors(target_image)
            if queries[0].shape[0] != self._matrix.shape[1]:
                raise RuntimeError("query and index descriptor dimensions differ")
            scores = np.max(np.stack([self._matrix @ query for query in queries]), axis=0)
            best_by_slug: dict[str, float] = {}
            for slug, score in zip(self._slugs, scores, strict=True):
                best_by_slug[slug] = max(best_by_slug.get(slug, -1.0), float(score))
            ranked = sorted(best_by_slug.items(), key=lambda pair: (-pair[1], pair[0]))
            first_slug, first_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            similarity = min(1.0, max(0.0, (first_score + 1.0) / 2.0))
            margin = min(1.0, max(0.0, first_score - second_score))
            confidence = min(1.0, similarity * 0.75 + margin * 0.25)
            confidence *= 0.75 + 0.25 * quality
            result = Match(
                self.catalog[first_slug], round(confidence, 4), round(margin, 4), quality_hint, True
            )

        # Validation shows complementary failure modes: DINOv2 is much stronger
        # for uncertain single-bottle views, while the narrow handcrafted centre
        # crop is stronger when edge-bearing objects flank the target.
        side_objects = False
        try:
            from .local_features import has_side_objects
            side_objects = has_side_objects(np.asarray(decoded.image))
        except ImportError:
            # The optional neural stage can still run without OpenCV; in that
            # reduced installation it cannot distinguish shelf layout here.
            pass
        if self._neural_features is not None and not side_objects:
            neural = self._neural_features.predict(decoded.image)
            # Cosine distance is retrieval evidence, not a calibrated probability.
            neural_confidence = min(0.86, max(0.50, 0.58 + 0.8 * neural["margin"]))
            result = Match(
                self.catalog[neural["slug"]],
                round(neural_confidence, 4),
                round(min(1.0, neural["margin"]), 4),
                quality_hint,
                True,
                "dinov2",
            )

        if self._ocr is not None:
            ocr_match = self._ocr.rank(target_image, self.settings.ocr_min_score)
            if ocr_match is not None and ocr_match.score >= self.settings.ocr_min_score:
                # OCR scores are lexical evidence, not probabilities. Saturating
                # them at 0.99 made a repeated generic token look certain. Use a
                # bounded curve and explicitly cap contradictory or alias-only
                # evidence instead.
                ocr_confidence = _ocr_confidence(
                    ocr_match, ocr_match.slug in self._indexed_slugs
                )
                result = Match(
                    self.catalog[ocr_match.slug],
                    round(ocr_confidence, 4),
                    round(min(1.0, ocr_match.margin / 10.0), 4),
                    quality_hint,
                    True,
                    "ocr",
                )

        self._remember(digest, result)
        return result

    def _remember(self, digest: str | None, result: Match) -> None:
        if digest and self.settings.cache_size > 0:
            with self._cache_lock:
                self._cache[digest] = result
                self._cache.move_to_end(digest)
                while len(self._cache) > self.settings.cache_size:
                    self._cache.popitem(last=False)

    @property
    def not_found_probability(self) -> float:
        fusion = self._recognizer.fusion if self._recognizer is not None else None
        if fusion is not None and fusion.not_found_threshold is not None:
            return float(fusion.not_found_threshold)
        return self.settings.not_found_probability

    @property
    def open_set_threshold(self) -> float:
        fusion = self._recognizer.fusion if self._recognizer is not None else None
        return fusion.open_set_threshold if fusion is not None else 0.5

    def image_path(self, slug: str) -> Path | None:
        wine = self.catalog.get(slug)
        directory = self.settings.images_dir
        if wine is None or not wine.photo_name or directory is None or slug in SCENE_PHOTO_SLUGS:
            return None
        path = (directory / wine.photo_name).resolve()
        if path.parent != directory.resolve() or not path.is_file():
            return None
        return path

    @staticmethod
    def sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()
