"""PP-OCRv5 label reading and catalogue-grounded text scoring.

The reader returns positioned words; the matcher turns them into sparse per-slug
evidence (token score, coverage, style/vintage agreement) that the fusion ranker
combines with visual similarity. Text never overrides the ranking on its own.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .models import Wine

# Letter and digit runs are separate tokens: OCR glues "КРАСНОЕ2022" together.
TOKEN_PATTERN = re.compile(r"[a-zа-яё]+|[0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"^(?:19[5-9]\d|20[0-4]\d)$")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")
ROMAN = re.compile(r"^[ivxlc]+$")
CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
})
# Latin glyphs that OCR emits for visually identical Cyrillic capitals.
LATIN_HOMOGLYPHS = str.maketrans({
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о",
    "p": "р", "t": "т", "x": "х", "y": "у", "3": "з",
})
HOMOGLYPH_ONLY = set("abcehkmoptxy3")
CYRILLIC = re.compile(r"[а-яё]")
# Labels print grape and style names in Latin script while the catalogue uses
# Russian (or the reverse). Each group maps to one canonical Cyrillic token.
WINE_TERMS = {
    "каберне": ("cabernet", "kaberne"), "совиньон": ("sauvignon", "sovinon"),
    "фран": ("franc", "fran"), "мерло": ("merlot", "merlo"), "пино": ("pinot", "pino"),
    "нуар": ("noir", "nuar"), "блан": ("blanc", "blancs", "blan"), "гри": ("gris",),
    "гриджио": ("grigio", "gridzhio"), "шардоне": ("chardonnay", "shardone"),
    "рислинг": ("riesling", "risling"), "сира": ("syrah", "shiraz", "sira", "шираз"),
    "мускат": ("muscat", "moscato", "muskat"), "алиготе": ("aligote",),
    "ркацители": ("rkatsiteli", "rkaciteli", "rkatsiteli"), "вионье": ("viognier", "vione"),
    "шенен": ("chenin", "shenen"), "траминер": ("traminer",), "мальбек": ("malbec", "malbek"),
    "санджовезе": ("sangiovese", "sandzhoveze"), "темпранильо": ("tempranillo",),
    "розе": ("rose", "roze"), "руж": ("rouge", "ruzh"), "брют": ("brut", "bryut"),
    "резерв": ("reserve", "reserva", "rezerv"), "совиньонблан": ("sauvignonblanc",),
    "саперави": ("saperavi",), "кокур": ("kokur",), "красностоп": ("krasnostop",),
    "цимлянский": ("tsimlyanskiy", "tsimlyansky"), "мюскадель": ("muscadelle",),
}
TERM_CANONICAL = {variant: canonical for canonical, variants in WINE_TERMS.items() for variant in variants}
STOP_TOKENS = {
    "вино", "wine", "vino", "винодельня", "winery", "россии", "russia", "год", "урожая",
    "сорт", "сорта", "винограда", "the", "and", "of", "de", "из", "для", "выдержка",
}
# Style words decide most near-duplicate pairs (same label, different sweetness).
STYLE_GROUPS = {
    "sweetness": {
        "brut_nature": ("брют натюр", "brut nature", "bryut natyur"),
        "extra_brut": ("экстра брют", "extra brut", "ekstra bryut"),
        "brut": ("брют", "brut", "bryut"),
        "semi_dry": ("полусухое", "polusuhoe", "semi dry", "demi sec"),
        "semi_sweet": ("полусладкое", "polusladkoe", "semi sweet"),
        "dry": ("сухое", "suhoe", "dry"),
        "sweet": ("сладкое", "sladkoe", "sweet", "десертное", "desertnoe"),
    },
    "colour": {
        "red": ("красное", "krasnoe", "red", "rosso"),
        "white": ("белое", "beloe", "white", "bianco", "byanko", "blanc"),
        "rose": ("розовое", "rozovoe", "rose", "rosé", "roze"),
        "orange": ("оранж", "oranzh", "orange"),
    },
}


def _roman_value(token: str) -> int | None:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    if not ROMAN.match(token) or len(token) < 2:
        return None
    total = 0
    for current, following in zip(token, token[1:] + " "):
        value = values[current]
        total += -value if following != " " and values.get(following, 0) > value else value
    return total if 1 < total < 200 else None


def normalize_tokens(text: str) -> list[str]:
    """Split, fold and expand text into comparable tokens (both scripts)."""
    text = CAMEL_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text))
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text.casefold().replace("ё", "е")):
        variants = {raw}
        if set(raw) <= HOMOGLYPH_ONLY and len(raw) >= 3:
            variants.add(raw.translate(LATIN_HOMOGLYPHS))
        elif CYRILLIC.search(raw):   # mixed script, e.g. "АРАТTИ"
            variants.add(raw.translate(LATIN_HOMOGLYPHS))
        for variant in tuple(variants):
            if variant in TERM_CANONICAL:
                variants.add(TERM_CANONICAL[variant])
        roman = _roman_value(raw)
        if roman is not None:
            variants.add(str(roman))
        for variant in tuple(variants):
            if any("а" <= ch <= "я" for ch in variant):
                variants.add(variant.translate(CYRILLIC_TO_LATIN))
        tokens.extend(sorted(variants))
    return [t for t in tokens if t not in STOP_TOKENS and len(t) >= (2 if t.isdigit() else 3)]


def style_of(text: str) -> dict[str, str]:
    folded = " " + " ".join(TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())) + " "
    found: dict[str, str] = {}
    for group, styles in STYLE_GROUPS.items():
        for style, phrases in styles.items():   # ordered: specific phrases first
            if any(f" {phrase} " in folded for phrase in phrases):
                found[group] = style
                break
    return found


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    score: float
    weight: float   # detector confidence x centrality x size prior


@dataclass(frozen=True, slots=True)
class TextEvidence:
    score: float
    coverage: float
    matched: tuple[str, ...]
    style_conflict: bool
    style_match: bool
    year_conflict: bool
    year_match: bool


class LabelReader:
    """PP-OCRv5 mobile detector + East-Slavic recogniser (Cyrillic and Latin)."""

    def __init__(self, model_dir: Path, max_side: int = 1280) -> None:
        from rapidocr import RapidOCR

        det = model_dir / "ch_PP-OCRv5_det_mobile.onnx"
        rec = model_dir / "eslav_PP-OCRv5_rec_mobile.onnx"
        for path in (det, rec):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.max_side = max_side
        self.engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,
            "Det.model_path": str(det),
            "Rec.model_path": str(rec),
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        })

    def read(self, image: Image.Image) -> list[Word]:
        image = image.convert("RGB")
        image.thumbnail((self.max_side, self.max_side), Image.Resampling.BILINEAR)
        width, height = image.size
        result = self.engine(np.asarray(image))
        if result.boxes is None:
            return []
        words = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            box = np.asarray(box, dtype=np.float32)
            cx, cy = box[:, 0].mean() / width, box[:, 1].mean() / height
            h = (box[:, 1].max() - box[:, 1].min()) / height
            # Neighbouring bottles sit at the frame edges: halve text weight at
            # the border. Larger glyphs (brand, wine name) carry more identity.
            centrality = math.exp(-((cx - 0.5) / 0.28) ** 2)
            size = min(1.0, 0.4 + h * 12)
            words.append(Word(text, float(score), float(score * (0.35 + 0.65 * centrality) * size)))
        return words


class TextMatcher:
    def __init__(self, catalog: dict[str, Wine]) -> None:
        self.slugs = sorted(catalog)
        self.fields: dict[str, dict[str, float]] = {}
        self.styles: dict[str, dict[str, str]] = {}
        self.years: dict[str, set[str]] = {}
        frequency: Counter[str] = Counter()
        for slug in self.slugs:
            wine = catalog[slug]
            weights: dict[str, float] = {}
            for text, weight in ((wine.name, 2.0), (wine.winery or "", 1.5), (wine.grapes or "", 0.8),
                                 (slug.replace("-", " "), 0.8), (wine.category or "", 0.5)):
                for token in normalize_tokens(text):
                    weights[token] = max(weights.get(token, 0.0), weight)
            self.fields[slug] = weights
            self.styles[slug] = style_of(f"{wine.name} {wine.category or ''} {slug.replace('-', ' ')}")
            self.years[slug] = {t for t in normalize_tokens(f"{wine.name} {slug}") if YEAR_PATTERN.match(t)}
            frequency.update(weights)
        total = len(self.slugs)
        self.idf = {t: math.log((total + 1) / (f + 1)) + 1.0 for t, f in frequency.items()}
        self.vocabulary = sorted(self.idf)
        self.postings: dict[str, list[str]] = {}
        for slug, weights in self.fields.items():
            for token in weights:
                self.postings.setdefault(token, []).append(slug)
        self.norms = {slug: sum(w * self.idf[t] for t, w in weights.items() if not t.isdigit())
                      for slug, weights in self.fields.items()}

    def _resolve(self, token: str) -> list[tuple[str, float]]:
        """Exact token, fuzzy nearest token, or a two-part compound split."""
        if token in self.idf:
            return [(token, 1.0)]
        if len(token) < 4 or token.isdigit():
            return []
        limit = 1 if len(token) <= 6 else 2
        found = process.extractOne(token, self.vocabulary, scorer=Levenshtein.distance, score_cutoff=limit)
        if found is not None and abs(len(found[0]) - len(token)) <= limit:
            return [(found[0], 0.6)]
        # OCR drops the space in "ПИНО НУАР" -> "ПИНОНУАР".
        for cut in range(3, len(token) - 2):
            left, right = token[:cut], token[cut:]
            if left in self.idf and right in self.idf:
                return [(left, 0.8), (right, 0.8)]
        return []

    def score(self, words: list[Word]) -> dict[str, TextEvidence]:
        # Each OCR token contributes once per slug, via its best-matching variant
        # (Cyrillic, transliterated, homoglyph-corrected or fuzzy).
        groups: dict[str, dict[str, float]] = {}
        full_text = " ".join(w.text for w in words if w.weight > 0.25)
        for word in words:
            for raw in TOKEN_PATTERN.findall(CAMEL_BOUNDARY.sub(" ", word.text)):
                group = groups.setdefault(raw.casefold(), {})
                for token in normalize_tokens(raw):
                    parts = self._resolve(token)
                    for part, (key, factor) in enumerate(parts):
                        # Halves of a split compound are separate words.
                        target = group if len(parts) == 1 else groups.setdefault(f"{raw.casefold()}#{part}", {})
                        target[key] = max(target.get(key, 0.0), word.weight * factor)
        query_style = style_of(full_text)
        query_years = {t for g in groups.values() for t in g if YEAR_PATTERN.match(t)}
        totals: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for raw, group in groups.items():
            best: dict[str, float] = {}
            for token, weight in group.items():
                for slug in self.postings.get(token, ()):
                    value = self.fields[slug][token] * self.idf[token] * weight
                    if value > best.get(slug, 0.0):
                        best[slug] = value
            for slug, value in best.items():
                totals[slug] = totals.get(slug, 0.0) + value
                # Record the OCR word itself so evidence is comparable across
                # candidates that matched different script variants of it.
                matched.setdefault(slug, []).append(raw)
        evidence = {}
        for slug, total in totals.items():
            style = self.styles[slug]
            conflict = any(g in style and style[g] != v for g, v in query_style.items())
            agree = any(style.get(g) == v for g, v in query_style.items())
            years = self.years[slug]
            evidence[slug] = TextEvidence(
                score=total,
                coverage=min(1.0, total / max(self.norms[slug], 1e-6)),
                matched=tuple(sorted(matched[slug])),
                style_conflict=conflict,
                style_match=agree and not conflict,
                year_conflict=bool(query_years and years and query_years.isdisjoint(years)),
                year_match=bool(query_years & years),
            )
        return evidence
