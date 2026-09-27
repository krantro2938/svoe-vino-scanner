"""Catalogue analogues: wines of the same style from other producers.

Used after a successful scan ("what else is like this?") and when the scanned
wine is not in the catalogue (show the closest style matches instead of a bare
"not found"). Similarity is explicit and explainable: colour, sweetness, grape
overlap and region, with visual similarity only as a tie-breaker.
"""
from __future__ import annotations

import re

from .label_text import style_of
from .models import Wine

GRAPE_SPLIT = re.compile(r"[,;/]+")


def grapes(wine: Wine) -> set[str]:
    return {g.strip().casefold() for g in GRAPE_SPLIT.split(wine.grapes or "") if g.strip()}


def wine_style(wine: Wine) -> dict[str, str]:
    return style_of(f"{wine.name} {wine.category or ''} {wine.slug.replace('-', ' ')}")


class AnalogFinder:
    def __init__(self, catalog: dict[str, Wine], visual=None) -> None:
        self.catalog = catalog
        self.visual = visual
        self.styles = {slug: wine_style(w) for slug, w in catalog.items()}
        self.grapes = {slug: grapes(w) for slug, w in catalog.items()}

    def _reasons(self, source: Wine, other: Wine) -> tuple[float, list[str]]:
        s, o = self.styles[source.slug], self.styles[other.slug]
        score, reasons = 0.0, []
        if s.get("colour") and s.get("colour") == o.get("colour"):
            score += 3.0; reasons.append("тот же цвет")
        elif (source.category or "") and source.category == other.category:
            score += 2.0; reasons.append("та же категория")
        if s.get("sweetness") and s.get("sweetness") == o.get("sweetness"):
            score += 2.0; reasons.append("та же сладость")
        shared = self.grapes[source.slug] & self.grapes[other.slug]
        if shared:
            union = self.grapes[source.slug] | self.grapes[other.slug]
            score += 3.0 * len(shared) / len(union)
            reasons.append("сорт: " + ", ".join(sorted(shared))[:60])
        if source.region and source.region == other.region:
            score += 1.0; reasons.append(f"регион {source.region}")
        return score, reasons

    def similar(self, slug: str, limit: int = 4, other_wineries: bool = True) -> list[tuple[Wine, list[str]]]:
        source = self.catalog[slug]
        visual = None
        if self.visual is not None and slug in self.visual.index_of:
            anchor = self.visual.views["bottle"][self.visual.index_of[slug]]
            visual = self.visual.views["bottle"] @ anchor
        ranked = []
        for other in self.catalog.values():
            if other.slug == slug or (other_wineries and other.winery and other.winery == source.winery):
                continue
            if other.slug not in self.styles:
                continue
            score, reasons = self._reasons(source, other)
            if visual is not None and other.slug in self.visual.index_of:
                score += 0.5 * float(visual[self.visual.index_of[other.slug]])
            if reasons:
                ranked.append((score, other.slug, other, reasons))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [(wine, reasons) for _, _, wine, reasons in ranked[:limit]]

    def closest_to_candidates(self, slugs: list[str], limit: int = 4) -> list[Wine]:
        """For an unrecognised label: the ranker's candidates, deduplicated."""
        seen, result = set(), []
        for slug in slugs:
            if slug in self.catalog and slug not in seen:
                seen.add(slug)
                result.append(self.catalog[slug])
        return result[:limit]

