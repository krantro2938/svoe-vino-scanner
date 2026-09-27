#!/usr/bin/env python3
"""Audit a generated catalogue without changing or guessing source mappings.

The report is intentionally deterministic and review-oriented.  It recomputes
index blockers, expands shared-photo conflicts to the affected product
identities, and checks known query aliases against the catalogue identity that
the alias targets.  Fuzzy token matching is used only to avoid reporting small
transliteration/spelling differences; it never changes catalogue records.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from scripts.build_catalog import CYRILLIC_TRANSLITERATION
except ModuleNotFoundError:  # Support ``python scripts/audit_catalog.py``.
    from build_catalog import CYRILLIC_TRANSLITERATION


AUDIT_SCHEMA_VERSION = 1
IDENTITY_FIELDS = ("name", "winery", "slug")
SHARED_PHOTO_FIELDS = ("name", "winery", "category", "grapes", "region")
TOKEN_RE = re.compile(r"[a-z0-9]+")
TOKEN_EQUIVALENTS = {
    "noir": "nuar",
    "pino": "pino",
    "pinot": "pino",
}


def normalize_query_text(value: str) -> str:
    """Return a stable Latin comparison form for Russian or Latin text."""
    translated = "".join(
        CYRILLIC_TRANSLITERATION.get(character.lower(), character.lower())
        for character in unicodedata.normalize("NFKC", value)
    )
    return " ".join(TOKEN_RE.findall(translated))


def query_tokens(value: str) -> tuple[str, ...]:
    """Tokenize query text while preserving first-seen order."""
    return tuple(dict.fromkeys(normalize_query_text(value).split()))


def _tokens_match(query: str, target: str) -> bool:
    query = TOKEN_EQUIVALENTS.get(query, query)
    target = TOKEN_EQUIVALENTS.get(target, target)
    if query == target:
        return True
    if query.isdigit() or target.isdigit() or min(len(query), len(target)) < 3:
        return False
    # This tolerates common catalogue transliterations such as pinot/pino and
    # noir/nuar, but is deliberately too strict to equate distinct short IDs.
    return difflib.SequenceMatcher(None, query, target, autojunk=False).ratio() >= 0.74


def unmatched_query_tokens(alias: str, record: Mapping[str, object]) -> list[str]:
    """Find alias terms absent from the record's query-visible identity."""
    surface = " ".join(str(record.get(field, "")) for field in IDENTITY_FIELDS)
    targets = query_tokens(surface)
    return [
        token
        for token in query_tokens(alias)
        if not any(_tokens_match(token, target) for target in targets)
    ]


def expected_index_blockers(record: Mapping[str, object], shared_photos: set[str]) -> list[str]:
    blockers: list[str] = []
    media_status = str(record.get("media_status", "unresolved"))
    if media_status != "resolved":
        blockers.append(f"media_{media_status}")
    photo_name = str(record.get("photo_name", ""))
    if photo_name and photo_name in shared_photos:
        blockers.append("shared_photo_across_slugs")
    return blockers


def _record_identity(record: Mapping[str, object]) -> dict[str, str]:
    return {
        "slug": str(record.get("slug", "")),
        "name": str(record.get("name", "")),
        "winery": str(record.get("winery", "")),
        "category": str(record.get("category", "")),
        "grapes": str(record.get("grapes", "")),
        "region": str(record.get("region", "")),
        "media_status": str(record.get("media_status", "")),
    }


def audit_catalog(
    records: Iterable[Mapping[str, object]],
    label_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Build an audit report from generated catalogue rows and known aliases."""
    rows = sorted(
        (dict(record) for record in records),
        key=lambda row: (
            str(row.get("slug", "")),
            str(row.get("photo_name", "")),
            str(row.get("name", "")),
        ),
    )
    by_slug: dict[str, list[dict]] = defaultdict(list)
    by_photo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_slug[str(row.get("slug", ""))].append(row)
        by_photo[str(row.get("photo_name", ""))].append(row)

    shared_photos = {
        photo
        for photo, photo_rows in by_photo.items()
        if photo and len({str(row.get("slug", "")) for row in photo_rows}) > 1
    }

    blocker_counts: Counter[str] = Counter()
    blocked_records: list[dict] = []
    metadata_inconsistencies: list[dict] = []
    for row in rows:
        expected = expected_index_blockers(row, shared_photos)
        blocker_counts.update(expected)
        if expected:
            blocked_records.append(
                {
                    "slug": str(row.get("slug", "")),
                    "photo_name": str(row.get("photo_name", "")),
                    "blockers": expected,
                }
            )
        declared = sorted(str(value) for value in row.get("index_blockers", []))
        declared_ready = row.get("index_ready")
        if declared != sorted(expected) or declared_ready != (not expected):
            metadata_inconsistencies.append(
                {
                    "slug": str(row.get("slug", "")),
                    "declared_index_ready": declared_ready,
                    "declared_blockers": declared,
                    "expected_index_ready": not expected,
                    "expected_blockers": expected,
                }
            )

    shared_photo_conflicts: list[dict] = []
    for photo_name in sorted(shared_photos):
        identities = sorted(
            (_record_identity(row) for row in by_photo[photo_name]),
            key=lambda row: (row["slug"], row["name"], row["winery"]),
        )
        differing_fields = [
            field
            for field in SHARED_PHOTO_FIELDS
            if len({identity[field].casefold() for identity in identities}) > 1
        ]
        shared_photo_conflicts.append(
            {
                "photo_name": photo_name,
                "record_count": len(identities),
                "differing_identity_fields": differing_fields,
                "records": identities,
            }
        )

    aliases = label_aliases or {}
    missing_targets: list[dict] = []
    ambiguous_targets: list[dict] = []
    aliases_with_identity_gaps: list[dict] = []
    normalized_alias_owners: dict[str, set[str]] = defaultdict(set)
    checked_aliases = 0
    for slug in sorted(aliases):
        values = aliases[slug]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Aliases for {slug!r} must be a list of strings")
        if not all(isinstance(alias, str) and alias.strip() for alias in values):
            raise ValueError(f"Aliases for {slug!r} must be non-empty strings")
        target_rows = by_slug.get(slug, [])
        if not target_rows:
            missing_targets.append({"slug": slug, "aliases": sorted(set(values))})
            continue
        if len(target_rows) != 1:
            ambiguous_targets.append(
                {"slug": slug, "record_count": len(target_rows), "aliases": sorted(set(values))}
            )
            continue
        record = target_rows[0]
        for alias in sorted(set(values), key=lambda value: (normalize_query_text(value), value)):
            checked_aliases += 1
            normalized_alias_owners[normalize_query_text(alias)].add(slug)
            missing = unmatched_query_tokens(alias, record)
            if missing:
                tokens = query_tokens(alias)
                aliases_with_identity_gaps.append(
                    {
                        "slug": slug,
                        "alias": alias,
                        "missing_tokens": missing,
                        "matched_token_count": len(tokens) - len(missing),
                        "token_count": len(tokens),
                        "catalog_identity": {
                            field: str(record.get(field, "")) for field in IDENTITY_FIELDS
                        },
                    }
                )

    alias_collisions = [
        {"normalized_alias": alias, "slugs": sorted(slugs)}
        for alias, slugs in sorted(normalized_alias_owners.items())
        if alias and len(slugs) > 1
    ]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "catalog": {
            "records": len(rows),
            "unique_nonempty_slugs": len({slug for slug in by_slug if slug}),
            "unique_nonempty_photo_names": len({photo for photo in by_photo if photo}),
            "records_ready": len(rows) - len(blocked_records),
            "records_blocked": len(blocked_records),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "blocked_records": blocked_records,
            "index_metadata_inconsistencies": metadata_inconsistencies,
        },
        "shared_photo_conflicts": {
            "group_count": len(shared_photo_conflicts),
            "record_count": sum(group["record_count"] for group in shared_photo_conflicts),
            "groups": shared_photo_conflicts,
        },
        "query_aliases": {
            "configured_slugs": len(aliases),
            "checked_aliases": checked_aliases,
            "missing_target_slugs": missing_targets,
            "ambiguous_target_slugs": ambiguous_targets,
            "aliases_with_identity_gaps": aliases_with_identity_gaps,
            "normalized_alias_collisions": alias_collisions,
        },
    }


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_label_aliases(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Label aliases must be a JSON object mapping slugs to string lists")
    result: dict[str, list[str]] = {}
    for slug, aliases in value.items():
        if not isinstance(slug, str) or not isinstance(aliases, list):
            raise ValueError("Label aliases must be a JSON object mapping slugs to string lists")
        if not all(isinstance(alias, str) and alias.strip() for alias in aliases):
            raise ValueError(f"Aliases for {slug!r} must be non-empty strings")
        result[slug] = aliases
    return result


def default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "data" / "generated" / "catalog" / "catalog.jsonl",
        root / "data" / "catalog" / "label_aliases.json",
    )


def make_parser() -> argparse.ArgumentParser:
    catalog, aliases = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=catalog)
    parser.add_argument("--label-aliases", type=Path, default=aliases)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        report = audit_catalog(load_jsonl(args.catalog), load_label_aliases(args.label_aliases))
        payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"catalog audit failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
