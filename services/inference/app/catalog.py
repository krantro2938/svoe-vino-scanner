from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .models import Wine


CSV_FIELDS = {
    "Название вина": "name",
    "Категория": "category",
    "Цвет": "color",
    "Регион": "region",
    "Сорт винограда": "grapes",
    "Описание": "description",
    "Винодельня": "winery",
    "Slug": "slug",
    "Название фото": "photo_name",
}

# The organiser's photo for these wines is an AI-generated table scene, not the
# bottle ("Generated Image June 22, 2026 - 10_44PM.webp"). Such a scene looks
# more like a random photo than any bottle shot, so it attracted non-wine
# queries; these wines are matched by label text only and shown without it.
SCENE_PHOTO_SLUGS = frozenset({"ona-skazala-da", "rozovoe-polusladkoe-2"})


def _clean(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"catalog line {line_number} is not an object")
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = value.get("items", value.get("wines", []))
    if not isinstance(value, list):
        raise ValueError("catalog JSON must be a list or contain an items/wines list")
    yield from value


def load_catalog(path: Path | None) -> dict[str, Wine]:
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "catalog not found; set CIFR_CATALOG_PATH to CSV, JSON, or JSONL"
        )

    rows: Iterable[dict[str, Any]]
    if path.suffix.casefold() == ".csv":
        source = path.open(encoding="utf-8-sig", newline="")
        rows = (
            {target: _clean(row.get(source_name)) for source_name, target in CSV_FIELDS.items()}
            for row in csv.DictReader(source)
        )
    else:
        source = None
        rows = ({key: _clean(value) for key, value in row.items()} for row in _iter_json(path))

    try:
        wines: dict[str, Wine] = {}
        for row in rows:
            slug = row.get("slug")
            name = row.get("name") or row.get("title")
            if not isinstance(slug, str) or not slug or not isinstance(name, str) or not name:
                continue
            normalized = dict(row)
            normalized["slug"] = slug
            normalized["name"] = name
            wines.setdefault(slug, Wine.model_validate(normalized))
    finally:
        if source is not None:
            source.close()

    if not wines:
        raise ValueError(f"catalog has no valid wine records: {path}")
    return wines
