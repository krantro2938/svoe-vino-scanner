"""Multi-signal recognition: visual shortlist, geometric and text verification, fusion.

1. SigLIP 2 view-to-view similarity scores the whole catalogue (shortlist).
2. PP-OCRv5 words are matched against catalogue text (sparse evidence).
3. RootSIFT + RANSAC counts geometric inliers for the union of shortlists.
4. A conditional-logit model trained on field-like photos turns the per-candidate
   features into a probability distribution over candidates.

The probability of the top candidate is the reported confidence; the gap to the
second candidate is the reported margin.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .label_text import TextEvidence, TextMatcher, Word

VISUAL_SHORTLIST = 30
TEXT_SHORTLIST = 10
FEATURES = (
    "v_max", "v_mean", "v_full_bottle", "v_centre_label", "v_gap", "v_rank",
    "sift_inliers", "sift_matches", "sift_best",
    "t_score", "t_gap", "t_coverage", "t_style_conflict", "t_style_match",
    "t_year_conflict", "t_year_match", "t_present", "t_missing",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    slug: str
    probability: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Recognition:
    candidates: tuple[Candidate, ...]
    ocr_text: str
    timings_ms: dict[str, float]
    in_catalogue: float | None = None

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    @property
    def margin(self) -> float:
        second = self.candidates[1].probability if len(self.candidates) > 1 else 0.0
        return self.best.probability - second


OPEN_SET_FEATURES = ("p1", "sift_inliers", "v_max", "t_coverage")


class FusionModel:
    def __init__(self, weights: dict[str, float], bias_note: str = "",
                 not_found_threshold: float | None = None,
                 open_set: dict[str, float] | None = None, open_set_threshold: float = 0.5) -> None:
        self.weights = weights
        self.note = bias_note
        # Calibrated by training/fit_fusion.py with simulated unknown wines.
        self.not_found_threshold = not_found_threshold
        # Logistic "is this wine in the catalogue at all?" model over the top
        # candidate's absolute evidence (training/fit_open_set.py). Softmax
        # probabilities are relative and stay high for an unknown wine that
        # merely beats weak competitors; absolute evidence does not.
        self.open_set = open_set
        self.open_set_threshold = open_set_threshold

    def in_catalogue(self, p1: float, features: dict[str, float]) -> float | None:
        if not self.open_set:
            return None
        values = {"p1": p1, **features}
        z = self.open_set.get("bias", 0.0) + sum(self.open_set.get(k, 0.0) * values.get(k, 0.0)
                                                  for k in OPEN_SET_FEATURES)
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: Path | None) -> "FusionModel":
        if path is not None and path.is_file():
            raw = json.loads(path.read_text())
            return cls({k: float(v) for k, v in raw["weights"].items()}, str(raw.get("trained_on", "")),
                       raw.get("not_found_threshold"), raw.get("open_set"),
                       float(raw.get("open_set_threshold", 0.5)))
        # Hand-set prior used only until a fitted model is available.
        return cls({"v_mean": 25.0, "v_gap": 10.0, "sift_inliers": 1.5, "sift_best": 2.0,
                    "t_score": 0.8, "t_gap": 0.8, "t_style_conflict": -2.0, "t_year_conflict": -2.0,
                    "t_style_match": 0.5, "t_year_match": 0.8}, "prior")

    def probabilities(self, rows: list[dict[str, float]]) -> np.ndarray:
        logits = np.array([sum(self.weights.get(k, 0.0) * v for k, v in row.items()) for row in rows])
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()


def candidate_features(
    slugs: list[str],
    visual: dict[str, np.ndarray],
    index_of: dict[str, int],
    text: dict[str, TextEvidence],
    sift: dict[str, dict],
    idf: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    """Per-candidate features relative to the best competitor in this query."""
    mean = visual["mean"]
    order = np.argsort(-mean)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    best_visual = float(mean[order[0]])
    best_text = max((e.score for e in text.values()), default=0.0)
    best_inliers = max((s["inliers"] for s in sift.values()), default=0)
    # Words on the label that some shortlisted wine explains: a sibling that
    # cannot explain e.g. "рислинг" should lose to the one that can.
    idf = idf or {}
    explained = {t for slug in slugs if slug in text for t in text[slug].matched}
    explained_weight = sum(idf.get(t, 1.0) for t in explained) or 1.0
    rows = []
    for slug in slugs:
        i = index_of[slug]
        ev = text.get(slug)
        sf = sift.get(slug, {"inliers": 0, "matches": 0})
        t_score = math.log1p(ev.score) if ev else 0.0
        rows.append({
            "v_max": float(visual["max"][i]),
            "v_mean": float(mean[i]),
            "v_full_bottle": float(visual["full>bottle"][i]),
            "v_centre_label": float(visual["centre>label"][i]),
            "v_gap": float(mean[i]) - best_visual,
            "v_rank": -math.log1p(float(rank[i])),
            "sift_inliers": math.log1p(sf["inliers"]),
            "sift_matches": math.log1p(sf["matches"]),
            "sift_best": float(sf["inliers"] >= 8 and sf["inliers"] == best_inliers),
            "t_score": t_score,
            "t_gap": t_score - math.log1p(best_text),
            "t_coverage": ev.coverage if ev else 0.0,
            "t_style_conflict": float(ev.style_conflict) if ev else 0.0,
            "t_style_match": float(ev.style_match) if ev else 0.0,
            "t_year_conflict": float(ev.year_conflict) if ev else 0.0,
            "t_year_match": float(ev.year_match) if ev else 0.0,
            "t_present": float(ev is not None),
            "t_missing": sum(idf.get(t, 1.0) for t in explained - set(ev.matched if ev else ()))
            / explained_weight if explained else 0.0,
        })
    return rows


def shortlist(visual: dict[str, np.ndarray], slugs: list[str], text: dict[str, TextEvidence]) -> list[str]:
    order = np.argsort(-visual["mean"])[:VISUAL_SHORTLIST]
    chosen = [slugs[i] for i in order]
    for slug in sorted(text, key=lambda s: -text[s].score)[:TEXT_SHORTLIST]:
        if slug not in chosen:
            chosen.append(slug)
    return chosen


class Recognizer:
    def __init__(self, catalog: dict, visual_dir: Path, ocr_dir: Path | None,
                 local_features=None, fusion_path: Path | None = None) -> None:
        from .encoder import VisualIndex

        self.visual = VisualIndex(visual_dir, catalog)
        self.text = TextMatcher({s: catalog[s] for s in self.visual.slugs})
        self.reader = None
        if ocr_dir is not None:
            from .label_text import LabelReader
            self.reader = LabelReader(ocr_dir)
        self.local = local_features
        self.fusion = FusionModel.load(fusion_path)
        self.pool = ThreadPoolExecutor(max_workers=2)

    def read_words(self, image: Image.Image) -> list[Word]:
        return self.reader.read(image) if self.reader is not None else []

    def verify(self, image: Image.Image, slugs: list[str]) -> dict[str, dict]:
        if self.local is None:
            return {}
        return self.local.verify(image, slugs)

    def rank(self, visual: dict[str, np.ndarray], words: list[Word], verify) -> tuple[list[str], list[dict], np.ndarray]:
        """Shortlist, verify and score; `verify(slugs)` returns RootSIFT evidence."""
        text = self.text.score(words)
        slugs = shortlist(visual, self.visual.slugs, text)
        sift = verify(slugs)
        rows = candidate_features(slugs, visual, self.visual.index_of, text, sift, self.text.idf)
        return slugs, rows, self.fusion.probabilities(rows)

    def recognize(self, image: Image.Image, words: list[Word] | None = None) -> Recognition:
        import time

        timings: dict[str, float] = {}
        started = time.perf_counter()
        ocr_future = None if words is not None else self.pool.submit(self.read_words, image)
        visual = self.visual.scores(self.visual.embed_query(image))
        timings["visual"] = (time.perf_counter() - started) * 1000
        if ocr_future is not None:
            words = ocr_future.result()
            timings["ocr"] = (time.perf_counter() - started) * 1000
        mark = time.perf_counter()
        slugs, rows, probabilities = self.rank(visual, words or [], lambda chosen: self.verify(image, chosen))
        timings["verify"] = (time.perf_counter() - mark) * 1000
        order = np.argsort(-probabilities)
        candidates = tuple(Candidate(slugs[i], float(probabilities[i]), rows[i]) for i in order)
        timings["total"] = (time.perf_counter() - started) * 1000
        in_catalogue = self.fusion.in_catalogue(candidates[0].probability, candidates[0].features)
        return Recognition(candidates, " ".join(w.text for w in words or []), timings, in_catalogue)
