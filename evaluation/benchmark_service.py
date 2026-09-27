#!/usr/bin/env python3
"""Sequentially smoke-test the recognition service and write evaluator JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    from .evaluate import InputError, _candidate_slugs, _first, _percentile, read_records
except ImportError:  # Direct script execution.
    from evaluate import InputError, _candidate_slugs, _first, _percentile, read_records


def multipart_image(image_path: Path) -> tuple[bytes, str]:
    boundary = "----cifr-" + uuid.uuid4().hex
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    safe_filename = image_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    body = prefix + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, boundary


def parse_response(body: bytes) -> tuple[str | None, list[str] | None]:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    if isinstance(value, list):
        candidates = _candidate_slugs({"candidates": value})
        slug = candidates[0] if candidates else None
        return slug, candidates
    if not isinstance(value, dict):
        return None, None
    slug = value.get("slug")
    if not isinstance(slug, str) or not slug.strip():
        slug = None
    else:
        slug = slug.strip()
    candidates = _candidate_slugs(value)
    if candidates is None and slug:
        candidates = [slug]
    return slug, candidates


def request_prediction(endpoint: str, image_path: Path, timeout: float) -> tuple[str | None, list[str] | None, int | None, str | None, float]:
    body, boundary = multipart_image(image_path)
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read(2 * 1024 * 1024 + 1)
            status = response.status
        if len(response_body) > 2 * 1024 * 1024:
            return None, None, status, "response exceeded 2 MiB", (time.perf_counter() - started) * 1000
        slug, candidates = parse_response(response_body) if status in (200, 201) else (None, None)
        error = None if slug else "successful response did not contain a valid slug"
        return slug, candidates, status, error, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return None, None, exc.code, f"HTTP {exc.code}", (time.perf_counter() - started) * 1000
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, None, None, str(exc), (time.perf_counter() - started) * 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/eval/predict")
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int, help="only send the first N manifest entries")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        print("ERROR: timeout and limit must be positive", file=sys.stderr)
        return 2
    if not args.images_dir.is_dir() or not args.manifest.is_file():
        print("ERROR: images directory or manifest not found", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        rows = read_records(args.manifest)
    except (InputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("ERROR: manifest has no rows", file=sys.stderr)
        return 2

    output_rows: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(rows, 1):
        query_id = _first(row, ("query_id",))
        relative = _first(row, ("image_path",))
        if not isinstance(query_id, str) or not query_id.strip() or not isinstance(relative, str) or not relative.strip():
            print(f"ERROR: invalid manifest row {index}", file=sys.stderr)
            return 2
        relative = relative.strip()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            print(f"ERROR: unsafe image_path in row {index}: {relative}", file=sys.stderr)
            return 2
        image_path = args.images_dir / relative_path
        try:
            image_path.resolve().relative_to(args.images_dir.resolve())
        except ValueError:
            print(f"ERROR: image_path escapes images directory: {relative}", file=sys.stderr)
            return 2
        if not image_path.is_file():
            print(f"ERROR: image not found: {image_path}", file=sys.stderr)
            return 2
        slug, candidates, status, error, latency = request_prediction(args.endpoint, image_path, args.timeout)
        result: dict[str, Any] = {
            "query_id": query_id.strip(),
            "image_path": relative,
            "image_sha256": sha256(image_path),
            "predicted_slug": slug,
            "latency_ms": round(latency),
        }
        if candidates and len(candidates) > 1:
            result["candidate_slugs"] = candidates
        output_rows.append(result)
        if error:
            failures += 1
            print(f"[{index}/{len(rows)}] {query_id}: {error} (status={status}, {latency:.0f} ms)", file=sys.stderr)
        else:
            print(f"[{index}/{len(rows)}] {query_id}: {slug} ({latency:.0f} ms)", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    temporary.replace(args.output)
    latencies = [row["latency_ms"] for row in output_rows]
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    print(f"Wrote {len(output_rows)} predictions to {args.output}; failures={failures}; p50={p50:.1f} ms; p95={p95:.1f} ms; max={max(latencies)} ms")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
