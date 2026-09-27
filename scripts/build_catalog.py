#!/usr/bin/env python3
"""Build deterministic catalog and Strapi-media manifests without extraction.

The script intentionally uses only Python's standard library.  It asks 7-Zip to
list the first volume of the multipart RAR; 7-Zip follows the remaining volumes
automatically.  No archive member is extracted by this importer.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
CSV_HEADERS = (
    "Название вина",
    "Категория",
    "Цвет",
    "Регион",
    "Сорт винограда",
    "Описание",
    "Винодельня",
    "Slug",
    "Название фото",
)
FIELD_NAMES = {
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
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}
RENDITION_RE = re.compile(r"^(?:(?:thumbnail|small|medium|large)_)+", re.I)
STRAPI_HASH_RE = re.compile(r"_[0-9a-f]{10}$", re.I)

# This spelling matches the forms visible in the supplied Strapi archive
# (for example, "Спуманте белый брют" -> "Spumante_belyj_bryut").
CYRILLIC_TRANSLITERATION = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "cz",
    "ч": "ch",
    "ш": "sh",
    "щ": "shh",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


@dataclass(frozen=True)
class ArchiveEntry:
    path: str
    size: int
    packed_size: int | None = None
    modified: str | None = None
    crc: str | None = None

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def extension(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def is_rendition(self) -> bool:
        return bool(RENDITION_RE.match(Path(self.filename).stem))


@dataclass(frozen=True)
class MediaMatch:
    status: str
    method: str | None
    selected: ArchiveEntry | None
    candidates: tuple[ArchiveEntry, ...]
    canonical_keys: tuple[str, ...]


def clean_value(value: str | None) -> str:
    """Normalize source whitespace while retaining paragraph boundaries."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def normalized_media_key(filename: str) -> str:
    """Return a conservative comparison key for an original upload name."""
    # Strapi drops the numero sign. NFKC would otherwise expand it to "No",
    # preventing an exact match for names such as "Бленд №4".
    stem = Path(os.path.basename(filename)).stem.replace("№", "")
    translated = "".join(
        CYRILLIC_TRANSLITERATION.get(char.lower(), char)
        for char in unicodedata.normalize("NFKC", stem)
    )
    return re.sub(r"[^a-z0-9]+", "_", translated.lower()).strip("_")


def archive_canonical_key(filename: str) -> str:
    """Remove a rendition marker and Strapi's ten-hex-character suffix."""
    stem = Path(os.path.basename(filename)).stem
    stem = RENDITION_RE.sub("", stem)
    stem = STRAPI_HASH_RE.sub("", stem)
    return normalized_media_key(stem)


def compact_media_key(key: str) -> str:
    """Ignore separator changes introduced by Strapi's case splitting."""
    return key.replace("_", "")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_7z_slt(output: str) -> list[ArchiveEntry]:
    """Parse the stable key/value form produced by ``7z l -slt``."""
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if current:
        records.append(current)

    entries: list[ArchiveEntry] = []
    for record in records:
        path = record.get("Path", "")
        if "/uploads/" not in path or record.get("Folder") == "+":
            continue
        try:
            size = int(record["Size"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Malformed 7-Zip record for {path!r}: missing numeric Size") from error
        packed = record.get("Packed Size")
        entries.append(
            ArchiveEntry(
                path=path,
                size=size,
                packed_size=int(packed) if packed and packed.isdigit() else None,
                modified=record.get("Modified") or None,
                crc=record.get("CRC") or None,
            )
        )
    return sorted(entries, key=lambda item: item.path)


def list_multipart_archive(first_volume: Path, seven_zip: str = "7z") -> tuple[list[ArchiveEntry], str]:
    executable = shutil.which(seven_zip)
    if not executable:
        raise RuntimeError(f"7-Zip executable {seven_zip!r} was not found")
    completed = subprocess.run(
        [executable, "l", "-slt", str(first_volume)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"7-Zip failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    entries = parse_7z_slt(completed.stdout)
    if not entries:
        raise RuntimeError("7-Zip returned no Strapi upload entries")
    diagnostics = "\n".join(
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if "warning" in line.lower()
        or "error" in line.lower()
        or "data after the end" in line.lower()
    )
    return entries, diagnostics


def load_catalog(csv_path: Path) -> tuple[list[dict], dict]:
    """Read, exact-deduplicate, clean, and audit the supplied CSV."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_HEADERS:
            raise ValueError(
                f"Unexpected CSV headers: {reader.fieldnames!r}; expected {list(CSV_HEADERS)!r}"
            )
        raw_rows = [(line_number, {key: row.get(key, "") for key in CSV_HEADERS}) for line_number, row in enumerate(reader, 2)]

    exact_groups: dict[tuple[str, ...], list[int]] = {}
    exact_rows: dict[tuple[str, ...], dict[str, str]] = {}
    for line_number, raw in raw_rows:
        key = tuple(raw[field] for field in CSV_HEADERS)
        exact_groups.setdefault(key, []).append(line_number)
        exact_rows.setdefault(key, raw)

    records: list[dict] = []
    for key, raw in exact_rows.items():
        cleaned = {FIELD_NAMES[field]: clean_value(raw[field]) for field in CSV_HEADERS}
        raw_overrides = {
            FIELD_NAMES[field]: raw[field]
            for field in CSV_HEADERS
            if raw[field] != cleaned[FIELD_NAMES[field]]
        }
        cleaned["source_rows"] = exact_groups[key]
        if raw_overrides:
            cleaned["raw_overrides"] = raw_overrides
        records.append(cleaned)
    records.sort(key=lambda row: (row["slug"], row["photo_name"], row["name"]))

    by_slug: dict[str, list[dict]] = defaultdict(list)
    by_photo: dict[str, set[str]] = defaultdict(set)
    by_title: dict[str, set[str]] = defaultdict(set)
    missing: dict[str, list[str]] = defaultdict(list)
    cleaned_fields: Counter[str] = Counter()
    for record in records:
        by_slug[record["slug"]].append(record)
        by_photo[record["photo_name"]].add(record["slug"])
        by_title[clean_value(record["name"]).casefold()].add(record["slug"])
        for field in FIELD_NAMES.values():
            if not record[field]:
                missing[field].append(record["slug"] or f"source-row-{record['source_rows'][0]}")
        cleaned_fields.update(record.get("raw_overrides", {}).keys())

    report = {
        "csv_data_rows": len(raw_rows),
        "exact_unique_rows": len(records),
        "exact_duplicate_rows_removed": len(raw_rows) - len(records),
        "exact_duplicate_groups": sum(len(lines) > 1 for lines in exact_groups.values()),
        "unique_nonempty_slugs": len({row["slug"] for row in records if row["slug"]}),
        "unique_nonempty_photo_names": len({row["photo_name"] for row in records if row["photo_name"]}),
        "cleaned_field_counts": dict(sorted(cleaned_fields.items())),
        "duplicate_slug_conflicts": [
            {
                "slug": slug,
                "records": [
                    {"name": row["name"], "photo_name": row["photo_name"], "source_rows": row["source_rows"]}
                    for row in rows
                ],
            }
            for slug, rows in sorted(by_slug.items())
            if slug and len(rows) > 1
        ],
        "photo_to_multiple_slugs": [
            {"photo_name": photo, "slugs": sorted(slugs)}
            for photo, slugs in sorted(by_photo.items())
            if photo and len(slugs) > 1
        ],
        "slug_to_multiple_photos": [
            {"slug": slug, "photo_names": sorted({row["photo_name"] for row in rows})}
            for slug, rows in sorted(by_slug.items())
            if slug and len({row["photo_name"] for row in rows}) > 1
        ],
        "normalized_title_to_multiple_slugs": [
            {"normalized_title": title, "slugs": sorted(slugs)}
            for title, slugs in sorted(by_title.items())
            if title and len(slugs) > 1
        ],
        "missing_fields": {
            field: {
                "count": len(slugs),
                "source_row_count": sum(
                    len(record["source_rows"])
                    for record in records
                    if not record[field]
                ),
                "slugs": sorted(slugs),
            }
            for field, slugs in sorted(missing.items())
        },
    }
    return records, report


def load_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError("Media overrides must be a JSON object mapping CSV photo names to archive paths")
    return payload


def build_media_indexes(entries: Sequence[ArchiveEntry]) -> tuple[dict[str, list[ArchiveEntry]], dict[str, set[str]], dict[str, ArchiveEntry]]:
    canonical: dict[str, list[ArchiveEntry]] = defaultdict(list)
    compact: dict[str, set[str]] = defaultdict(set)
    by_path: dict[str, ArchiveEntry] = {}
    for entry in entries:
        by_path[entry.path] = entry
        if entry.extension not in IMAGE_EXTENSIONS:
            continue
        key = archive_canonical_key(entry.filename)
        canonical[key].append(entry)
        compact[compact_media_key(key)].add(key)
    return canonical, compact, by_path


def _select_best(candidates: Iterable[ArchiveEntry]) -> ArchiveEntry:
    choices = list(candidates)
    originals = [entry for entry in choices if not entry.is_rendition]
    pool = originals or choices
    return sorted(pool, key=lambda item: (-item.size, item.path))[0]


def reconcile_media(
    photo_name: str,
    canonical: Mapping[str, Sequence[ArchiveEntry]],
    compact: Mapping[str, set[str]],
    by_path: Mapping[str, ArchiveEntry],
    overrides: Mapping[str, str],
) -> MediaMatch:
    if photo_name in overrides:
        override_path = overrides[photo_name]
        if override_path not in by_path:
            raise ValueError(f"Override for {photo_name!r} points to absent archive path {override_path!r}")
        selected = by_path[override_path]
        return MediaMatch("resolved", "override", selected, (selected,), (archive_canonical_key(selected.filename),))

    query_key = normalized_media_key(photo_name)
    keys: tuple[str, ...]
    method: str
    if query_key in canonical:
        keys = (query_key,)
        method = "normalized_name"
    elif (embedded_extension_key := f"{query_key}_{Path(photo_name).suffix.lower().lstrip('.')}") in canonical:
        # Some uploads retained the source extension as part of the Strapi name
        # before being transcoded, e.g. source.png -> source_png_<hash>.webp.
        keys = (embedded_extension_key,)
        method = "normalized_name_with_embedded_extension"
    else:
        compact_keys = sorted(compact.get(compact_media_key(query_key), set()))
        if len(compact_keys) != 1:
            return MediaMatch("unresolved", None, None, (), tuple(compact_keys))
        keys = (compact_keys[0],)
        method = "separator_insensitive_name"

    candidates = tuple(sorted((entry for key in keys for entry in canonical[key]), key=lambda item: item.path))
    selected = _select_best(candidates)
    original_uploads = [entry for entry in candidates if not entry.is_rendition]
    # Reuploads with the same byte size and archive CRC are interchangeable.
    # Different fingerprints sharing one source name remain genuinely ambiguous.
    fingerprints = {(entry.size, entry.crc) for entry in original_uploads}
    status = "ambiguous" if len(fingerprints) > 1 else "resolved"
    return MediaMatch(status, method, selected, candidates, keys)


def media_suggestions(
    photo_name: str,
    canonical: Mapping[str, Sequence[ArchiveEntry]],
    limit: int = 3,
) -> list[dict]:
    """Return review-only fuzzy suggestions; never silently select one."""
    query = normalized_media_key(photo_name)
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, query, key, autojunk=False).ratio(), key)
            for key in canonical
        ),
        key=lambda item: (-item[0], item[1]),
    )[:limit]
    return [
        {
            "similarity": round(score, 4),
            "canonical_key": key,
            "suggested_path": _select_best(canonical[key]).path,
        }
        for score, key in scored
    ]


def archive_summary(entries: Sequence[ArchiveEntry], volumes: Sequence[Path], diagnostics: str) -> dict:
    extensions = Counter(entry.extension or "<none>" for entry in entries)
    images = [entry for entry in entries if entry.extension in IMAGE_EXTENSIONS]
    return {
        "schema_version": SCHEMA_VERSION,
        "archive_volumes": [path.name for path in volumes],
        "volume_bytes": sum(path.stat().st_size for path in volumes),
        "upload_entries": len(entries),
        "upload_uncompressed_bytes": sum(entry.size for entry in entries),
        "image_entries": len(images),
        "original_image_entries": sum(not entry.is_rendition for entry in images),
        "rendition_image_entries": sum(entry.is_rendition for entry in images),
        "extensions": dict(sorted(extensions.items())),
        "seven_zip_diagnostics": diagnostics.splitlines() if diagnostics else [],
    }


def json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text("".join(json_line(row) + "\n" for row in rows), encoding="utf-8")


def source_manifest(paths: Sequence[Path], include_hashes: bool) -> list[dict]:
    return [
        {
            # Source files are siblings in the supplied dataset. Basenames keep
            # the generated manifest byte-for-byte stable across workstations.
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if include_hashes else None,
        }
        for path in paths
    ]


def build(args: argparse.Namespace) -> dict:
    csv_path = args.csv.resolve()
    first_volume = args.archive_first.resolve()
    volumes = sorted(first_volume.parent.glob("prod-svoe-vino-strapi.part*.rar"))
    if not volumes:
        raise FileNotFoundError(f"No multipart RAR volumes found beside {first_volume}")
    if first_volume not in volumes:
        raise ValueError(f"First archive volume {first_volume} is outside the discovered volume set")

    records, csv_report = load_catalog(csv_path)
    entries, diagnostics = list_multipart_archive(first_volume, args.seven_zip)
    canonical, compact, by_path = build_media_indexes(entries)
    overrides = load_overrides(args.overrides.resolve() if args.overrides else None)

    slugs_by_photo: dict[str, set[str]] = defaultdict(set)
    for record in records:
        slugs_by_photo[record["photo_name"]].add(record["slug"])

    media_rows: list[dict] = []
    matches_by_photo: dict[str, MediaMatch] = {}
    for photo_name in sorted(slugs_by_photo):
        match = reconcile_media(photo_name, canonical, compact, by_path, overrides)
        matches_by_photo[photo_name] = match
        row = {
            "photo_name": photo_name,
            "slugs": sorted(slugs_by_photo[photo_name]),
            "status": match.status,
            "method": match.method,
            "selected": asdict(match.selected) if match.selected else None,
            "candidate_count": len(match.candidates),
        }
        originals = [entry for entry in match.candidates if not entry.is_rendition]
        row["original_candidate_count"] = len(originals)
        row["distinct_original_fingerprint_count"] = len(
            {(entry.size, entry.crc) for entry in originals}
        )
        if match.status == "ambiguous":
            row["candidate_originals"] = [
                entry.path for entry in match.candidates if not entry.is_rendition
            ]
        elif match.status == "unresolved":
            row["review_suggestions"] = media_suggestions(photo_name, canonical)
        media_rows.append(row)

    catalog_rows = []
    index_blocker_counts: Counter[str] = Counter()
    for record in records:
        match = matches_by_photo[record["photo_name"]]
        row = dict(record)
        row["media_status"] = match.status
        row["media_path"] = match.selected.path if match.status == "resolved" else None
        row["media_crc32"] = match.selected.crc if match.status == "resolved" else None
        if match.status == "ambiguous" and match.selected:
            row["media_review_candidate_path"] = match.selected.path
        blockers = []
        if match.status != "resolved":
            blockers.append(f"media_{match.status}")
        if len(slugs_by_photo[record["photo_name"]]) > 1:
            blockers.append("shared_photo_across_slugs")
        row["index_ready"] = not blockers
        if blockers:
            row["index_blockers"] = blockers
            index_blocker_counts.update(blockers)
        catalog_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_report = archive_summary(entries, volumes, diagnostics)
    status_counts = Counter(match.status for match in matches_by_photo.values())
    method_counts = Counter(match.method or "none" for match in matches_by_photo.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "sources": source_manifest([csv_path, *volumes], not args.skip_source_hashes),
        "csv": csv_report,
        "archive": archive_report,
        "media": {
            "photo_names": len(matches_by_photo),
            "status_counts": dict(sorted(status_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
            "unresolved": [
                {
                    "photo_name": photo,
                    "review_suggestions": media_suggestions(photo, canonical),
                }
                for photo, match in sorted(matches_by_photo.items())
                if match.status == "unresolved"
            ],
            "ambiguous": [photo for photo, match in sorted(matches_by_photo.items()) if match.status == "ambiguous"],
            "override_count": sum(match.method == "override" for match in matches_by_photo.values()),
        },
        "index_readiness": {
            "records_ready": sum(row["index_ready"] for row in catalog_rows),
            "records_blocked": sum(not row["index_ready"] for row in catalog_rows),
            "blocker_counts": dict(sorted(index_blocker_counts.items())),
        },
    }
    write_jsonl(output_dir / "catalog.jsonl", catalog_rows)
    write_jsonl(output_dir / "media_manifest.jsonl", media_rows)
    write_json(output_dir / "archive_inventory.json", archive_report)
    write_json(output_dir / "catalog_import_report.json", report)

    if csv_report["duplicate_slug_conflicts"] and not args.allow_slug_conflicts:
        raise RuntimeError(
            "Divergent records share a slug; reports were written, but the import is invalid. "
            "Inspect catalog_import_report.json or rerun with --allow-slug-conflicts."
        )
    return report


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "csv": root / "Датасет" / "strapi_output0709.csv",
        "archive_first": root / "Датасет" / "prod-svoe-vino-strapi.part1.rar",
        "overrides": root / "data" / "catalog" / "media_overrides.json",
        "output_dir": root / "data" / "generated" / "catalog",
    }


def make_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=defaults["csv"])
    parser.add_argument("--archive-first", type=Path, default=defaults["archive_first"])
    parser.add_argument("--overrides", type=Path, default=defaults["overrides"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--seven-zip", default="7z")
    parser.add_argument("--skip-source-hashes", action="store_true", help="Faster local iteration; emits null hashes")
    parser.add_argument("--allow-slug-conflicts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        report = build(args)
    except (FileNotFoundError, RuntimeError, ValueError, csv.Error, json.JSONDecodeError) as error:
        print(f"catalog import failed: {error}", file=sys.stderr)
        return 2
    counts = report["media"]["status_counts"]
    print(
        f"catalog import complete: {report['csv']['exact_unique_rows']} records; "
        f"media resolved={counts.get('resolved', 0)}, ambiguous={counts.get('ambiguous', 0)}, "
        f"unresolved={counts.get('unresolved', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
