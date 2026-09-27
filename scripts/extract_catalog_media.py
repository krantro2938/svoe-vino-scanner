#!/usr/bin/env python3
"""Selectively extract resolved catalog media from the multipart Strapi RAR."""

from __future__ import annotations

import argparse
import binascii
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ExtractionItem:
    photo_name: str
    archive_path: str
    size: int
    crc32: str
    slugs: tuple[str, ...]

    @property
    def index_ready(self) -> bool:
        return len(self.slugs) == 1


def validate_photo_name(name: str) -> str:
    """Allow one literal filename, never a path or control sequence."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError("photo_name must be a non-empty filename")
    if "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"unsafe photo_name contains a path separator: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(f"unsafe photo_name contains a control character: {name!r}")
    if len(os.fsencode(name)) > 240:
        raise ValueError(f"photo_name leaves insufficient room for a safe staging suffix: {name!r}")
    return name


def validate_archive_path(value: str) -> str:
    """Require a relative POSIX member path with no traversal components."""
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"unsafe archive member path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {value!r}")
    return value


def load_extraction_plan(manifest: Path, limit: int | None = None) -> list[ExtractionItem]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")
    items: list[ExtractionItem] = []
    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on manifest line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            if row.get("status") != "resolved":
                continue
            selected = row.get("selected")
            if not isinstance(selected, dict):
                raise ValueError(f"resolved manifest line {line_number} has no selected object")
            photo_name = validate_photo_name(row.get("photo_name"))
            archive_path = validate_archive_path(selected.get("path"))
            try:
                size = int(selected["size"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"resolved manifest line {line_number} has an invalid size") from error
            if size < 0:
                raise ValueError(f"resolved manifest line {line_number} has a negative size")
            crc32 = selected.get("crc")
            if not isinstance(crc32, str) or len(crc32) != 8:
                raise ValueError(f"resolved manifest line {line_number} has an invalid CRC32")
            try:
                int(crc32, 16)
            except ValueError as error:
                raise ValueError(f"resolved manifest line {line_number} has an invalid CRC32") from error
            slugs = row.get("slugs")
            if not isinstance(slugs, list) or not slugs or not all(isinstance(slug, str) and slug for slug in slugs):
                raise ValueError(f"resolved manifest line {line_number} has invalid slugs")
            items.append(
                ExtractionItem(
                    photo_name=photo_name,
                    archive_path=archive_path,
                    size=size,
                    crc32=crc32.upper(),
                    slugs=tuple(sorted(set(slugs))),
                )
            )

    items.sort(key=lambda item: (item.photo_name, item.archive_path))
    duplicate_names = [
        name
        for name, count in _counts(item.photo_name for item in items).items()
        if count > 1
    ]
    if duplicate_names:
        raise ValueError(f"resolved manifest contains duplicate photo_name values: {duplicate_names[:3]!r}")
    return items[:limit] if limit is not None else items


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def crc32_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    checksum = 0
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            checksum = binascii.crc32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08X}"


def verify_file(path: Path, item: ExtractionItem) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == item.size
        and crc32_file(path) == item.crc32
    )


def _safe_extracted_source(root: Path, archive_path: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(archive_path).parts)
    if root.resolve() not in candidate.resolve().parents:
        raise ValueError(f"archive member escaped temporary extraction directory: {archive_path!r}")
    return candidate


def _install_verified(source: Path, destination: Path, item: ExtractionItem, overwrite: bool) -> str:
    if destination.is_symlink():
        raise ValueError(f"refusing to use symlink destination: {destination}")
    if verify_file(destination, item):
        return "already_verified"
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"destination exists but does not match manifest; use --overwrite after review: {destination}"
        )
    staged = destination.with_name(f".{destination.name}.extracting")
    if staged.exists() or staged.is_symlink():
        raise FileExistsError(f"staging path already exists: {staged}")
    try:
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
        if not verify_file(staged, item):
            raise ValueError(f"extracted bytes fail size/CRC verification for {item.photo_name!r}")
        os.replace(staged, destination)
    finally:
        if staged.exists() and not staged.is_symlink():
            staged.unlink()
    return "extracted"


def write_index_manifest(path: Path, items: Sequence[ExtractionItem]) -> None:
    rows = [
        {
            "photo_name": item.photo_name,
            "local_path": item.photo_name,
            "source_archive_path": item.archive_path,
            "size": item.size,
            "crc32": item.crc32,
            "slugs": list(item.slugs),
            "index_ready": item.index_ready,
        }
        for item in items
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def extract(
    items: Sequence[ExtractionItem],
    archive_first: Path,
    output_dir: Path,
    index_manifest: Path,
    seven_zip: str,
    overwrite: bool,
) -> dict[str, int]:
    executable = shutil.which(seven_zip)
    if not executable:
        raise RuntimeError(f"7-Zip executable {seven_zip!r} was not found")
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [item for item in items if not verify_file(output_dir / item.photo_name, item)]
    by_archive_path: dict[str, list[ExtractionItem]] = defaultdict(list)
    for item in pending:
        by_archive_path[item.archive_path].append(item)

    results = {"selected": len(items), "extracted": 0, "already_verified": len(items) - len(pending)}
    if pending:
        with tempfile.TemporaryDirectory(prefix=".catalog-extract-", dir=output_dir.parent) as directory:
            temporary_root = Path(directory)
            listfile = temporary_root / "members.txt"
            listfile.write_text("".join(path + "\n" for path in sorted(by_archive_path)), encoding="utf-8")
            completed = subprocess.run(
                [
                    executable,
                    "x",
                    "-y",
                    "-spd",
                    "-scsUTF-8",
                    f"-o{temporary_root}",
                    str(archive_first),
                    f"@{listfile}",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode not in (0, 1):
                raise RuntimeError(
                    f"7-Zip selective extraction failed with exit code {completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )
            for archive_path, destinations in sorted(by_archive_path.items()):
                source = _safe_extracted_source(temporary_root, archive_path)
                if not source.is_file():
                    raise FileNotFoundError(f"7-Zip did not extract requested member: {archive_path}")
                representative = destinations[0]
                if source.stat().st_size != representative.size or crc32_file(source) != representative.crc32:
                    raise ValueError(f"archive member failed manifest size/CRC verification: {archive_path}")
                for item in destinations:
                    status = _install_verified(source, output_dir / item.photo_name, item, overwrite)
                    results[status] += 1

    write_index_manifest(index_manifest, items)
    return results


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "manifest": root / "data" / "generated" / "catalog" / "media_manifest.jsonl",
        "archive_first": root / "Датасет" / "prod-svoe-vino-strapi.part1.rar",
        "output_dir": root / "data" / "catalog-images",
        "index_manifest": root / "data" / "generated" / "catalog" / "extracted_media.jsonl",
    }


def make_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=defaults["manifest"])
    parser.add_argument("--archive-first", type=Path, default=defaults["archive_first"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--index-manifest", type=Path, default=defaults["index_manifest"])
    parser.add_argument("--seven-zip", default="7z")
    parser.add_argument("--limit", type=int, help="Extract the first N resolved names, sorted deterministically")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing or invoking 7-Zip")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing file only after extracting and CRC-verifying its replacement")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        items = load_extraction_plan(args.manifest.resolve(), args.limit)
        unique_members = len({item.archive_path for item in items})
        if args.dry_run:
            print(
                f"dry run: {len(items)} resolved photo names, {unique_members} unique archive members, "
                f"{sum(item.size for item in items)} bytes; no files written"
            )
            return 0
        if not items:
            raise ValueError("manifest contains no resolved media to extract")
        results = extract(
            items,
            args.archive_first.resolve(),
            args.output_dir.resolve(),
            args.index_manifest.resolve(),
            args.seven_zip,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"catalog media extraction failed: {error}", file=sys.stderr)
        return 2
    print(
        f"selective extraction complete: selected={results['selected']}, "
        f"extracted={results['extracted']}, already_verified={results['already_verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
