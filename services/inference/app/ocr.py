from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .models import Wine


CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
)
ROMAN_NUMBERS = {"xviii": "18", "xxiv": "24", "xxxvi": "36", "xlv": "45"}
TOKEN_PATTERN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")

# These occur in many winery names and are not useful evidence of a producer by
# themselves.  Keeping them out of the conflict check prevents a generic word
# such as "winery" from vetoing an otherwise strong label-name match.
GENERIC_PRODUCER_TOKENS = {
    "estate",
    "vino",
    "wine",
    "winery",
    "вино",
    "винодельня",
    "завод",
    "поместье",
}

# OCR often invents short colour/style words from decorative Cyrillic text
# (for example ``ред`` -> ``red``). They occur on too many labels to identify a
# bottle and can otherwise outweigh a genuine product-family token.
NON_DISTINCTIVE_LABEL_TOKENS = {
    "red",
    "rose",
    "vino",
    "white",
    "wine",
    "белое",
    "вино",
    "красное",
    "ред",
    "розовое",
}


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    values = {
        token
        for token in TOKEN_PATTERN.findall(normalized)
        if len(token) >= 3 and (not token.isdigit() or len(token) >= 4 or token in {"18", "24", "36", "45"})
    }
    expanded = set(values)
    for token in values:
        expanded.add(token.translate(CYRILLIC_TO_LATIN))
        if token in ROMAN_NUMBERS:
            expanded.add(ROMAN_NUMBERS[token])
        if len(token) == 3 and token.startswith("0") and token.isdigit():
            expanded.add("2" + token)
    expanded.update(token.translate(CYRILLIC_TO_LATIN) for token in tuple(expanded))
    return {token for token in expanded if token and token not in NON_DISTINCTIVE_LABEL_TOKENS}


@dataclass(frozen=True, slots=True)
class OCRCandidate:
    slug: str
    score: float
    matched_tokens: tuple[str, ...]
    producer_conflict: bool = False
    vintage_conflict: bool = False
    alias_dependent: bool = False


@dataclass(frozen=True, slots=True)
class OCRMatch:
    slug: str
    score: float
    margin: float
    text: str
    candidates: tuple[OCRCandidate, ...] = ()
    producer_conflict: bool = False
    vintage_conflict: bool = False
    alias_dependent: bool = False


class LabelOCR:
    """Local Tesseract label reader with catalog-grounded lexical ranking."""

    def __init__(
        self,
        catalog: dict[str, Wine],
        tessdata_dir: Path,
        aliases_path: Path | None = None,
        workers: int = 6,
    ) -> None:
        executable = shutil.which("tesseract")
        if not executable:
            raise RuntimeError("tesseract executable is not installed")
        if not (tessdata_dir / "eng.traineddata").is_file():
            raise FileNotFoundError("eng.traineddata is missing")
        if not (tessdata_dir / "rus.traineddata").is_file():
            raise FileNotFoundError("Russian Tesseract model is missing")

        self.executable = executable
        self.tessdata_dir = tessdata_dir
        self.workers = max(1, workers)
        aliases: dict[str, list[str]] = {}
        if aliases_path and aliases_path.is_file():
            value = json.loads(aliases_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                aliases = {
                    slug: [alias for alias in raw if isinstance(alias, str)]
                    for slug, raw in value.items()
                    if isinstance(raw, list)
                }

        self.fields: dict[str, list[tuple[set[str], float]]] = {}
        self.alias_tokens: dict[str, set[str]] = {}
        self.canonical_tokens: dict[str, set[str]] = {}
        self.winery_tokens: dict[str, set[str]] = {}
        self.candidate_years: dict[str, set[str]] = {}
        document_frequency: Counter[str] = Counter()
        producer_frequency: Counter[str] = Counter()
        for slug, wine in catalog.items():
            slug_aliases = _tokens(" ".join(aliases.get(slug, [])))
            name_tokens = _tokens(wine.name)
            winery_tokens = _tokens(wine.winery or "")
            other_tokens = _tokens(" ".join(filter(None, [wine.grapes, wine.slug, wine.photo_name])))
            weighted = [
                (name_tokens | slug_aliases, 2.0),
                (winery_tokens, 1.5),
                (other_tokens, 1.0),
            ]
            self.fields[slug] = weighted
            self.alias_tokens[slug] = slug_aliases
            self.canonical_tokens[slug] = name_tokens | winery_tokens | other_tokens
            self.winery_tokens[slug] = winery_tokens
            canonical_text = " ".join(
                filter(None, [wine.name, wine.slug, wine.photo_name])
            ).casefold()
            self.candidate_years[slug] = set(YEAR_PATTERN.findall(canonical_text))
            document_frequency.update(set().union(*(tokens for tokens, _ in weighted)))
            producer_frequency.update(winery_tokens)
        total = len(catalog)
        self.idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        # A producer token is considered identifying only when it occurs in at
        # most 1% of the catalog. This allows evidence such as "Massandra" to
        # catch a conflict without treating generic winery vocabulary as one.
        producer_limit = max(3, math.ceil(total * 0.01))
        self.producer_tokens = {
            token
            for token, frequency in producer_frequency.items()
            if frequency <= producer_limit and token not in GENERIC_PRODUCER_TOKENS
        }

    @staticmethod
    def _crop(image: Image.Image, box: tuple[float, float, float, float], angle: int) -> Image.Image:
        width, height = image.size
        left, top, right, bottom = box
        cropped = image.crop(
            (int(left * width), int(top * height), int(right * width), int(bottom * height))
        )
        if angle:
            cropped = cropped.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        if cropped.width > 1600:
            cropped.thumbnail((1600, 2000), Image.Resampling.LANCZOS)
        return ImageOps.autocontrast(ImageOps.grayscale(cropped))

    def _read_one(self, task: tuple[Path, int, str, float, float]) -> str:
        path, psm, language, center_left, center_right = task
        completed = subprocess.run(
            [
                self.executable,
                str(path),
                "stdout",
                "--tessdata-dir",
                str(self.tessdata_dir),
                "-l",
                language,
                "--psm",
                str(psm),
                "-c",
                "tessedit_create_tsv=1",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        if completed.returncode != 0:
            return ""
        return self._central_words(completed.stdout, center_left, center_right)

    @staticmethod
    def _central_words(tsv: str, center_left: float, center_right: float) -> str:
        """Keep complete words whose centers lie on the target bottle column."""
        words = []
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
            try:
                midpoint = float(row["left"]) + float(row["width"]) / 2
                confidence = float(row["conf"])
            except (KeyError, ValueError, TypeError):
                continue
            text = row.get("text", "").strip()
            if text and confidence >= 0 and center_left <= midpoint <= center_right:
                words.append(text)
        return " ".join(words)

    @staticmethod
    def _regions() -> tuple[
        tuple[tuple[float, float, float, float], int, int, str], ...
    ]:
        return (
            ((0.15, 0.30, 0.72, 0.78), -3, 6, "rus_best"),
            ((0.15, 0.30, 0.72, 0.78), 0, 6, "rus_best"),
            ((0.15, 0.30, 0.72, 0.78), 4, 6, "rus_best"),
            ((0.18, 0.10, 0.82, 0.90), 0, 6, "rus"),
            ((0.18, 0.10, 0.82, 0.90), 0, 11, "eng"),
            ((0.15, 0.22, 0.85, 0.48), 0, 11, "eng"),
            ((0.15, 0.38, 0.85, 0.66), 0, 11, "eng"),
            ((0.15, 0.55, 0.85, 0.82), 0, 11, "rus"),
        )

    def _read_regions(
        self,
        prepared: Image.Image,
        directory: Path,
        indexes: tuple[int, ...],
    ) -> str:
        regions = self._regions()
        tasks: list[tuple[Path, int, str, float, float]] = []
        for index in indexes:
            box, angle, psm, language = regions[index]
            path = directory / f"region-{index}.png"
            cropped = self._crop(prepared, box, angle)
            cropped.save(path, format="PNG", optimize=False)
            # Map the central 40% of the source into this OCR region. Rotation
            # is at most four degrees; using the rotated width keeps this a
            # conservative spatial prior without cutting words before reading.
            left, _, right, _ = box
            center_left = (0.30 - left) / (right - left) * cropped.width
            center_right = (0.70 - left) / (right - left) * cropped.width
            tasks.append((path, psm, language, center_left, center_right))
        with ThreadPoolExecutor(max_workers=min(self.workers, len(tasks))) as executor:
            return "\n".join(executor.map(self._read_one, tasks))

    def read(self, image: Image.Image) -> str:
        """Read every OCR region, primarily for diagnostics."""
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        with tempfile.TemporaryDirectory(prefix="cifr-ocr-") as directory:
            return self._read_regions(
                prepared, Path(directory), tuple(range(len(self._regions())))
            )

    def _fuzzy_map(self, query: set[str]) -> dict[str, str]:
        """Map unseen OCR tokens to their nearest catalogue token, if close enough."""
        vocabulary = getattr(self, "_vocabulary", None)
        if vocabulary is None:
            vocabulary = sorted(
                {token for fields in self.fields.values() for tokens, _ in fields for token in tokens}
            )
            self._vocabulary = vocabulary
        known = set(vocabulary)
        mapping: dict[str, str] = {}
        for token in query:
            if token in known or len(token) < 4 or token.isdigit():
                continue
            limit = 1 if len(token) <= 6 else 2
            found = process.extractOne(
                token, vocabulary, scorer=Levenshtein.distance, score_cutoff=limit
            )
            if found is not None and abs(len(found[0]) - len(token)) <= limit:
                mapping[token] = found[0]
        return mapping

    def rank_text(self, text: str) -> OCRMatch | None:
        query = _tokens(text)
        if not query:
            return None
        ranked: list[OCRCandidate] = []
        producer_evidence = query & getattr(self, "producer_tokens", set())
        query_years = set(YEAR_PATTERN.findall(" ".join(query)))
        fuzzy_tokens = self._fuzzy_map(query)
        for slug, weighted_fields in self.fields.items():
            score = 0.0
            matched_tokens: set[str] = set()
            for token in query:
                exact_weight = max(
                    (weight for tokens, weight in weighted_fields if token in tokens),
                    default=0.0,
                )
                if exact_weight:
                    score += exact_weight * self.idf.get(token, 1.0)
                    matched_tokens.add(token)
                    continue
                # OCR on curved, glossy labels substitutes or drops glyphs
                # ("DONLM", "МУСКАТЕДЛ"). Accept the closest catalogue token
                # within a small, length-dependent edit distance at reduced
                # weight instead of hand-listing individual misreadings.
                fuzzy = fuzzy_tokens.get(token)
                if fuzzy:
                    fuzzy_weight = max(
                        (weight for tokens, weight in weighted_fields if fuzzy in tokens),
                        default=0.0,
                    )
                    if fuzzy_weight:
                        score += 0.6 * fuzzy_weight * self.idf.get(fuzzy, 1.0)
                        matched_tokens.add(fuzzy)
            if score:
                winery_tokens = getattr(self, "winery_tokens", {}).get(slug, set())
                candidate_years = getattr(self, "candidate_years", {}).get(slug, set())
                alias_tokens = getattr(self, "alias_tokens", {}).get(slug, set())
                canonical_tokens = getattr(self, "canonical_tokens", {}).get(
                    slug, set().union(*(tokens for tokens, _ in weighted_fields))
                )
                ranked.append(
                    OCRCandidate(
                        slug=slug,
                        score=round(score, 4),
                        matched_tokens=tuple(sorted(matched_tokens)),
                        producer_conflict=bool(producer_evidence)
                        and not bool(query & winery_tokens),
                        vintage_conflict=bool(query_years and candidate_years)
                        and query_years.isdisjoint(candidate_years),
                        alias_dependent=bool((query & alias_tokens) - canonical_tokens),
                    )
                )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item.score, item.slug))
        winner = ranked[0]
        second = ranked[1].score if len(ranked) > 1 else 0.0
        return OCRMatch(
            winner.slug,
            winner.score,
            round(max(0.0, winner.score - second), 4),
            text,
            tuple(ranked[:5]),
            winner.producer_conflict,
            winner.vintage_conflict,
            winner.alias_dependent,
        )

    def rank(self, image: Image.Image, minimum_score: float = 0.0) -> OCRMatch | None:
        """Rank center-filtered words; extra regions never admit side-label text."""
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        primary_indexes = (2, 3, 6)
        secondary_indexes = (0, 1, 4, 5, 7)
        with tempfile.TemporaryDirectory(prefix="cifr-ocr-") as directory:
            temp_directory = Path(directory)
            primary_text = self._read_regions(prepared, temp_directory, primary_indexes)
            primary_match = self.rank_text(primary_text)
            if primary_match is not None and primary_match.score >= minimum_score:
                return primary_match
            secondary_text = self._read_regions(prepared, temp_directory, secondary_indexes)
        return self.rank_text(f"{primary_text}\n{secondary_text}")
