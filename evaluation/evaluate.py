#!/usr/bin/env python3
"""Validate scanner predictions and calculate reproducible retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


LABEL_FIELDS = ("expected_slug", "true_slug", "ground_truth_slug", "label", "slug")
CANDIDATE_FIELDS = ("candidate_slugs", "top_5", "top5", "top_k", "candidates", "matches")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
QUERY_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class InputError(ValueError):
    """Raised when an input file cannot be parsed safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise InputError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        if path.suffix.lower() not in {".tsv", ".csv"}:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
            except csv.Error:
                pass
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}: invalid JSON: {exc.msg}") from exc
        if isinstance(value, dict):
            for key in ("items", "catalog", "queries", "predictions", "records"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise InputError(f"{path}: JSON must be an array of objects (or contain one)")
        return value
    return _read_table(path)


def _first(row: dict[str, Any], names: Iterable[str]) -> Any:
    lower = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and value != "":
            return value
    return None


def load_catalog_slugs(path: Path) -> set[str]:
    rows = read_records(path)
    slugs: set[str] = set()
    for number, row in enumerate(rows, 2):
        value = _first(row, ("slug", "Slug"))
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise InputError(f"{path}:{number}: catalog slug must be a non-empty string")
        slugs.add(value.strip())
    if not slugs:
        raise InputError(f"{path}: no slug column/values found")
    return slugs


def _issue(issues: list[dict[str, Any]], level: str, code: str, message: str, **context: Any) -> None:
    item = {"level": level, "code": code, "message": message}
    if context:
        item["context"] = context
    issues.append(item)


def _label(row: dict[str, Any]) -> str | None:
    value = _first(row, LABEL_FIELDS)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _candidate_slugs(row: dict[str, Any]) -> list[str] | None:
    value = _first(row, CANDIDATE_FIELDS)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return []
        elif stripped:
            value = [part.strip() for part in re.split(r"[|,]", stripped) if part.strip()]
        else:
            value = []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        slug = item if isinstance(item, str) else item.get("slug") if isinstance(item, dict) else None
        if isinstance(slug, str) and slug.strip() and slug.strip() not in result:
            result.append(slug.strip())
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
            if not match:
                raise InputError(f"{path}:{number}: invalid SHA-256 checksum line")
            values[match.group(2).replace("\\", "/").lstrip("./")] = match.group(1).lower()
    return values


def _checksum_for(checksums: dict[str, str], image_path: str) -> str | None:
    normalized = image_path.replace("\\", "/").lstrip("./")
    exact = checksums.get(normalized)
    if exact:
        return exact
    suffix_matches = [value for key, value in checksums.items() if key.endswith("/" + normalized)]
    return suffix_matches[0] if len(set(suffix_matches)) == 1 else None


def _percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _binary_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y"}


def _slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["predicted_slug"] == row["expected_slug"] for row in rows)
    candidate_rows = [row for row in rows if row["candidate_slugs"] is not None]
    hits = sum(row["expected_slug"] in row["candidate_slugs"][:5] for row in candidate_rows)
    return {
        "queries": len(rows),
        "classes": len({row["expected_slug"] for row in rows}),
        "accuracy_at_1": _rounded(correct / len(rows)) if rows else None,
        "candidate_coverage": len(candidate_rows),
        "recall_at_5": _rounded(hits / len(candidate_rows)) if candidate_rows else None,
    }


def _grouped_metrics(labeled: list[dict[str, Any]]) -> dict[str, Any] | None:
    field = next(
        (
            candidate
            for candidate in ("hard_negative_group", "group_id", "group")
            if any(row.get(candidate) is not None for row in labeled)
        ),
        None,
    )
    if field is None:
        explicit_hard = [row for row in labeled if row.get("is_hard_negative")]
        return {"explicit_hard_negative": _slice_summary(explicit_hard)} if explicit_hard else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in labeled:
        value = row.get(field)
        if value is not None:
            grouped.setdefault(str(value), []).append(row)
    per_group = {name: _slice_summary(rows) for name, rows in sorted(grouped.items())}
    confusable = {
        name: rows for name, rows in grouped.items()
        if len({row["expected_slug"] for row in rows}) >= 2
    }
    hard_rows = [row for rows in confusable.values() for row in rows]
    result: dict[str, Any] = {
        "field": field,
        "group_count": len(grouped),
        "per_group": per_group,
        "macro_group_accuracy_at_1": _rounded(
            sum(summary["accuracy_at_1"] for summary in per_group.values()) / len(per_group)
        ) if per_group else None,
    }
    if hard_rows:
        result["hard_negative"] = {
            "definition": "rows in groups containing at least two ground-truth slugs",
            "group_count": len(confusable),
            **_slice_summary(hard_rows),
        }
    explicit_hard = [row for row in labeled if row.get("is_hard_negative")]
    if explicit_hard:
        result["explicit_hard_negative"] = _slice_summary(explicit_hard)
    return result


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    labeled = [row for row in rows if row["expected_slug"] is not None]
    if not labeled:
        return None
    correct = sum(row["predicted_slug"] == row["expected_slug"] for row in labeled)
    classes = sorted(
        {row["expected_slug"] for row in labeled}
        | {row["predicted_slug"] for row in labeled if row["predicted_slug"] is not None}
    )
    per_class: dict[str, dict[str, float | int]] = {}
    class_f1_values: list[float] = []
    for slug in classes:
        tp = sum(row["expected_slug"] == slug and row["predicted_slug"] == slug for row in labeled)
        fp = sum(row["expected_slug"] != slug and row["predicted_slug"] == slug for row in labeled)
        fn = sum(row["expected_slug"] == slug and row["predicted_slug"] != slug for row in labeled)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1_values.append(f1)
        per_class[slug] = {
            "support": sum(row["expected_slug"] == slug for row in labeled),
            "precision": _rounded(precision),
            "recall": _rounded(recall),
            "f1": _rounded(f1),
        }
    accuracy = correct / len(labeled)
    candidate_rows = [row for row in labeled if row["candidate_slugs"] is not None]
    recall_at_5 = None
    mrr = None
    if candidate_rows:
        hits = 0
        reciprocal_ranks = 0.0
        for row in candidate_rows:
            top_five = row["candidate_slugs"][:5]
            if row["expected_slug"] in top_five:
                hits += 1
                reciprocal_ranks += 1.0 / (top_five.index(row["expected_slug"]) + 1)
        recall_at_5 = hits / len(candidate_rows)
        mrr = reciprocal_ranks / len(candidate_rows)
    metrics = {
        "labeled_queries": len(labeled),
        "correct_top1": correct,
        "accuracy_at_1": _rounded(accuracy),
        "micro_f1": _rounded(accuracy),
        "macro_f1": _rounded(sum(class_f1_values) / len(class_f1_values)),
        "candidate_coverage": len(candidate_rows),
        "recall_at_5": _rounded(recall_at_5),
        "mrr_at_5": _rounded(mrr),
        "per_class": per_class,
    }
    grouped = _grouped_metrics(labeled)
    if grouped is not None:
        metrics["grouped"] = grouped
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for row in labeled:
        if row.get("split") is not None:
            split_rows.setdefault(str(row["split"]), []).append(row)
    if split_rows:
        metrics["by_split"] = {
            name: _slice_summary(split) for name, split in sorted(split_rows.items())
        }
    return metrics


def evaluate(
    query_manifest: Path,
    predictions_path: Path,
    *,
    catalog_path: Path | None = None,
    images_dir: Path | None = None,
    checksums_path: Path | None = None,
    candidates_path: Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    queries_raw = read_records(query_manifest)
    predictions_raw = read_records(predictions_path)
    catalog_slugs = load_catalog_slugs(catalog_path) if catalog_path else None
    checksums = _load_checksums(checksums_path) if checksums_path else {}

    queries: dict[str, dict[str, Any]] = {}
    query_order: list[str] = []
    for number, row in enumerate(queries_raw, 2):
        query_id = _first(row, ("query_id",))
        image_path = _first(row, ("image_path",))
        if not isinstance(query_id, str) or not QUERY_ID_RE.fullmatch(query_id.strip()):
            _issue(issues, "error", "invalid_query_id", "query_id must use only letters, digits, dot, underscore, or hyphen", row=number)
            continue
        query_id = query_id.strip()
        if query_id in queries:
            _issue(issues, "error", "duplicate_query_id", "query manifest contains a duplicate query_id", query_id=query_id)
            continue
        if not isinstance(image_path, str) or not image_path.strip():
            _issue(issues, "error", "invalid_image_path", "image_path must be a non-empty string", query_id=query_id)
            continue
        path_obj = Path(image_path)
        if path_obj.is_absolute() or ".." in path_obj.parts:
            _issue(issues, "error", "unsafe_image_path", "image_path must be a safe relative path", query_id=query_id)
        expected = _label(row)
        declared_checksum = _first(row, ("image_sha256", "sha256", "checksum"))
        if declared_checksum is not None and (
            not isinstance(declared_checksum, str) or not SHA256_RE.fullmatch(declared_checksum)
        ):
            _issue(issues, "error", "invalid_query_checksum", "query checksum must contain 64 hexadecimal characters", query_id=query_id)
            declared_checksum = None
        elif isinstance(declared_checksum, str):
            declared_checksum = declared_checksum.lower()
        if expected and catalog_slugs is not None and expected not in catalog_slugs:
            _issue(issues, "error", "unknown_expected_slug", "ground-truth slug is absent from the catalog", query_id=query_id, slug=expected)
        queries[query_id] = {
            "image_path": image_path.strip(),
            "expected_slug": expected,
            "image_sha256": declared_checksum,
            "split": _first(row, ("split",)),
            "hard_negative_group": _first(row, ("hard_negative_group", "label_family", "confusion_group")),
            "group_id": _first(row, ("group_id",)),
            "group": _first(row, ("group",)),
            "is_hard_negative": _binary_flag(_first(row, ("is_hard_negative", "hard_negative"))),
        }
        query_order.append(query_id)

    if not queries:
        _issue(issues, "error", "empty_query_manifest", "query manifest contains no valid rows")
    if catalog_slugs is None:
        _issue(issues, "info", "catalog_not_provided", "Catalog validation was skipped because --catalog was not provided.")

    candidate_map: dict[str, list[str]] = {}
    if candidates_path:
        for number, row in enumerate(read_records(candidates_path), 1):
            query_id = _first(row, ("query_id",))
            candidates = _candidate_slugs(row)
            if not isinstance(query_id, str) or not QUERY_ID_RE.fullmatch(query_id.strip()):
                _issue(issues, "error", "invalid_candidate_query_id", "candidate row has no valid query_id", row=number)
            elif candidates is None:
                _issue(issues, "error", "missing_candidates", "candidate row has no supported candidate list", query_id=query_id)
            else:
                normalized_query_id = query_id.strip()
                if normalized_query_id in candidate_map:
                    _issue(issues, "error", "duplicate_candidate_query_id", "candidate file contains a duplicate query_id", query_id=normalized_query_id)
                else:
                    candidate_map[normalized_query_id] = candidates
        for query_id in sorted(set(candidate_map) - set(queries)):
            _issue(issues, "error", "extra_candidate_query_id", "candidate query_id is absent from manifest", query_id=query_id)

    predictions: dict[str, dict[str, Any]] = {}
    prediction_order: list[str] = []
    latencies: list[float] = []
    for number, row in enumerate(predictions_raw, 1):
        missing_fields = []
        for required_field in ("query_id", "image_path", "image_sha256", "predicted_slug", "latency_ms"):
            if required_field not in row:
                missing_fields.append(required_field)
                _issue(
                    issues,
                    "error",
                    "missing_prediction_field",
                    "prediction row is missing a required field",
                    row=number,
                    field=required_field,
                )
        query_id = row.get("query_id")
        image_path = row.get("image_path")
        image_sha256 = row.get("image_sha256")
        slug = row.get("predicted_slug")
        latency = row.get("latency_ms")
        valid_row = not missing_fields
        if not isinstance(query_id, str) or not QUERY_ID_RE.fullmatch(query_id.strip()):
            _issue(issues, "error", "invalid_prediction_query_id", "prediction query_id must use only letters, digits, dot, underscore, or hyphen", row=number)
            valid_row = False
            query_id = f"__invalid_row_{number}"
        else:
            query_id = query_id.strip()
        if query_id in predictions:
            _issue(issues, "error", "duplicate_prediction", "predictions contain a duplicate query_id", query_id=query_id)
            continue
        if not isinstance(image_path, str) or not image_path:
            _issue(issues, "error", "invalid_prediction_image_path", "prediction image_path must be a non-empty string", query_id=query_id)
            valid_row = False
        if not isinstance(image_sha256, str) or not SHA256_RE.fullmatch(image_sha256):
            _issue(issues, "error", "invalid_prediction_checksum", "image_sha256 must contain 64 hexadecimal characters", query_id=query_id)
            valid_row = False
        else:
            image_sha256 = image_sha256.lower()
        if slug is not None and (not isinstance(slug, str) or not slug.strip()):
            _issue(issues, "error", "invalid_predicted_slug", "predicted_slug must be null or a non-empty string", query_id=query_id)
            valid_row = False
        elif isinstance(slug, str):
            slug = slug.strip()
            if catalog_slugs is not None and slug not in catalog_slugs:
                _issue(issues, "error", "unknown_predicted_slug", "predicted slug is absent from the catalog", query_id=query_id, slug=slug)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            _issue(issues, "error", "invalid_latency", "latency_ms must be a finite non-negative number", query_id=query_id)
            valid_row = False
        else:
            latency = float(latency)
        candidates = candidate_map.get(query_id, _candidate_slugs(row))
        if candidates is not None and catalog_slugs is not None:
            for candidate in candidates:
                if candidate not in catalog_slugs:
                    _issue(issues, "error", "unknown_candidate_slug", "candidate slug is absent from the catalog", query_id=query_id, slug=candidate)
        predictions[query_id] = {
            "image_path": image_path,
            "image_sha256": image_sha256,
            "predicted_slug": slug,
            "latency_ms": latency,
            "candidate_slugs": candidates,
            "schema_valid": valid_row,
        }
        prediction_order.append(query_id)

    query_ids = set(queries)
    prediction_ids = set(predictions)
    for query_id in sorted(query_ids - prediction_ids):
        _issue(issues, "error", "missing_prediction", "query has no prediction", query_id=query_id)
    for query_id in sorted(prediction_ids - query_ids):
        _issue(issues, "error", "extra_prediction", "prediction query_id is absent from manifest", query_id=query_id)
    common_order = [query_id for query_id in prediction_order if query_id in query_ids]
    if common_order != [query_id for query_id in query_order if query_id in prediction_ids]:
        _issue(issues, "warning", "prediction_order", "prediction order differs from query manifest order")

    scored_rows: list[dict[str, Any]] = []
    checksum_matches = 0
    checksum_checked = 0
    for query_id in query_order:
        query = queries[query_id]
        prediction = predictions.get(query_id)
        if prediction is None:
            scored_rows.append({
                "expected_slug": query["expected_slug"],
                "predicted_slug": None,
                "candidate_slugs": None,
                "split": query["split"],
                "hard_negative_group": query["hard_negative_group"],
                "group_id": query["group_id"],
                "group": query["group"],
                "is_hard_negative": query["is_hard_negative"],
            })
            continue
        if prediction["image_path"] != query["image_path"]:
            _issue(issues, "error", "image_path_mismatch", "prediction image_path differs from manifest", query_id=query_id)
        if isinstance(prediction["latency_ms"], float) and math.isfinite(prediction["latency_ms"]) and prediction["latency_ms"] >= 0:
            latencies.append(prediction["latency_ms"])
        actual_checksum: str | None = None
        if images_dir:
            image_file = images_dir / query["image_path"]
            if not image_file.is_file():
                _issue(issues, "error", "image_missing", "query image does not exist", query_id=query_id, path=str(image_file))
            else:
                actual_checksum = _sha256(image_file)
                checksum_checked += 1
                if prediction["image_sha256"] == actual_checksum:
                    checksum_matches += 1
                else:
                    _issue(issues, "error", "prediction_checksum_mismatch", "prediction checksum does not match image bytes", query_id=query_id)
                declared = _checksum_for(checksums, query["image_path"]) if checksums else None
                if checksums and declared is None:
                    _issue(issues, "warning", "checksum_not_declared", "image is absent from checksum manifest", query_id=query_id)
                elif declared is not None and declared != actual_checksum:
                    _issue(issues, "error", "source_checksum_mismatch", "image bytes do not match checksum manifest", query_id=query_id)
                if query["image_sha256"] is not None and query["image_sha256"] != actual_checksum:
                    _issue(issues, "error", "query_checksum_mismatch", "image bytes do not match checksum declared in query manifest", query_id=query_id)
        else:
            declared = query["image_sha256"] or (_checksum_for(checksums, query["image_path"]) if checksums else None)
            if declared is not None:
                checksum_checked += 1
                if prediction["image_sha256"] == declared:
                    checksum_matches += 1
                else:
                    _issue(issues, "error", "prediction_checksum_mismatch", "prediction checksum does not match checksum manifest", query_id=query_id)
        scored_rows.append({
            "expected_slug": query["expected_slug"],
            "predicted_slug": prediction["predicted_slug"],
            "candidate_slugs": prediction["candidate_slugs"],
            "split": query["split"],
            "hard_negative_group": query["hard_negative_group"],
            "group_id": query["group_id"],
            "group": query["group"],
            "is_hard_negative": query["is_hard_negative"],
        })
        if prediction["candidate_slugs"] and prediction["predicted_slug"] != prediction["candidate_slugs"][0]:
            _issue(issues, "warning", "candidate_top1_mismatch", "first candidate does not equal predicted_slug", query_id=query_id)

    labeled_count = sum(row["expected_slug"] is not None for row in scored_rows)
    if not labeled_count:
        _issue(issues, "info", "unlabeled_manifest", "No ground-truth labels were found; quality metrics are intentionally omitted. This is submission-integrity validation only.")
    elif labeled_count != len(scored_rows):
        _issue(issues, "warning", "partial_labels", "Quality metrics cover only manifest rows that contain labels", labeled=labeled_count, matched=len(scored_rows))
    if scored_rows and not checksum_checked:
        _issue(issues, "warning", "checksums_not_verified", "No source image or declared checksum was available; prediction checksums were format-checked only.")

    classification = _classification_metrics(scored_rows)
    valid_known = sum(
        prediction["predicted_slug"] is not None
        and (catalog_slugs is None or prediction["predicted_slug"] in catalog_slugs)
        for query_id, prediction in predictions.items() if query_id in query_ids
    )
    matched_predictions = [prediction for query_id, prediction in predictions.items() if query_id in query_ids]
    errors = sum(issue["level"] == "error" for issue in issues)
    warnings = sum(issue["level"] == "warning" for issue in issues)
    return {
        "report_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "scored" if classification else "unlabeled_validation",
        "status": "invalid" if errors else "valid",
        "inputs": {
            "query_manifest": str(query_manifest),
            "predictions": str(predictions_path),
            "catalog": str(catalog_path) if catalog_path else None,
            "images_dir": str(images_dir) if images_dir else None,
            "checksums": str(checksums_path) if checksums_path else None,
            "candidates": str(candidates_path) if candidates_path else None,
        },
        "validation": {
            "query_count": len(queries),
            "prediction_count": len(predictions_raw),
            "matched_count": len(query_ids & prediction_ids),
            "missing_predictions": len(query_ids - prediction_ids),
            "extra_predictions": len(prediction_ids - query_ids),
            "null_predictions": sum(prediction["predicted_slug"] is None for query_id, prediction in predictions.items() if query_id in query_ids),
            "non_null_predictions": sum(prediction["predicted_slug"] is not None for prediction in matched_predictions),
            "catalog_slug_count": len(catalog_slugs) if catalog_slugs is not None else None,
            "known_non_null_predictions": valid_known,
            "predictions_with_candidates": sum(prediction["candidate_slugs"] is not None for prediction in matched_predictions),
            "checksums_checked": checksum_checked,
            "checksums_matched": checksum_matches,
            "error_count": errors,
            "warning_count": warnings,
            "schema_valid_predictions": sum(
                prediction["schema_valid"] for query_id, prediction in predictions.items() if query_id in query_ids
            ),
        },
        "metrics": classification,
        "latency_ms": {
            "count": len(latencies),
            "p50": _rounded(_percentile(latencies, 0.50)),
            "p95": _rounded(_percentile(latencies, 0.95)),
            "max": _rounded(max(latencies) if latencies else None),
        },
        "issues": issues,
    }


def markdown_report(report: dict[str, Any]) -> str:
    validation = report["validation"]
    latency = report["latency_ms"]
    lines = [
        "# Recognition evaluation report",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Mode: `{report['mode']}`",
        f"- Queries / predictions / matched: {validation['query_count']} / {validation['prediction_count']} / {validation['matched_count']}",
        f"- Validation errors / warnings: {validation['error_count']} / {validation['warning_count']}",
        f"- Null predictions: {validation['null_predictions']}",
        f"- Schema-valid matched predictions: {validation['schema_valid_predictions']} / {validation['matched_count']}",
        f"- Catalog-known non-null predictions: {validation['known_non_null_predictions']} / {validation['non_null_predictions']}",
        f"- Predictions with ranked candidates: {validation['predictions_with_candidates']}",
        f"- Checksum matches: {validation['checksums_matched']} / {validation['checksums_checked']}",
        "",
        "## Quality",
        "",
    ]
    metrics = report["metrics"]
    if metrics is None:
        lines.append("Ground-truth labels are absent. Accuracy and F1 are not computable; this report validates submission integrity only.")
    else:
        lines.extend([
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Labeled queries | {metrics['labeled_queries']} |",
            f"| Accuracy@1 | {metrics['accuracy_at_1']:.3f} |",
            f"| Micro-F1 | {metrics['micro_f1']:.3f} |",
            f"| Macro-F1 | {metrics['macro_f1']:.3f} |",
            f"| Recall@5 | {metrics['recall_at_5'] if metrics['recall_at_5'] is not None else 'N/A'} |",
            f"| Top-5 candidate coverage | {metrics['candidate_coverage']} |",
        ])
        grouped = metrics.get("grouped")
        if grouped:
            hard = grouped.get("hard_negative") or grouped.get("explicit_hard_negative")
            lines.extend(["", "### Group slices", ""])
            if grouped.get("field"):
                lines.append(
                    f"Grouped by `{grouped['field']}` across {grouped['group_count']} groups; "
                    f"macro group Accuracy@1: {grouped['macro_group_accuracy_at_1']}."
                )
            if hard:
                lines.append(
                    f"Hard-negative slice: {hard['queries']} queries, Accuracy@1 "
                    f"{hard['accuracy_at_1']}, Recall@5 {hard['recall_at_5']}."
                )
    lines.extend([
        "",
        "## Latency",
        "",
        "| Requests | p50 (ms) | p95 (ms) | max (ms) |",
        "| ---: | ---: | ---: | ---: |",
        f"| {latency['count']} | {latency['p50']} | {latency['p95']} | {latency['max']} |",
        "",
        "## Validation findings",
        "",
    ])
    if not report["issues"]:
        lines.append("No findings.")
    else:
        for issue in report["issues"]:
            context = issue.get("context", {})
            suffix = "" if not context else " — " + ", ".join(f"{key}={value}" for key, value in context.items())
            lines.append(f"- **{issue['level'].upper()} `{issue['code']}`:** {issue['message']}{suffix}")
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path, help="TSV/CSV/JSON(L) query manifest")
    parser.add_argument("--predictions", required=True, type=Path, help="predictions JSONL")
    parser.add_argument("--catalog", type=Path, help="catalog CSV or generated JSON(L) manifest")
    parser.add_argument("--images-dir", type=Path, help="root directory for query images")
    parser.add_argument("--checksums", type=Path, help="GNU-style SHA-256 manifest")
    parser.add_argument("--candidates", type=Path, help="optional JSON(L)/table with Top-K candidate lists")
    parser.add_argument("--report-json", type=Path, default=Path("evaluation-report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("evaluation-report.md"))
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(
            args.queries,
            args.predictions,
            catalog_path=args.catalog,
            images_dir=args.images_dir,
            checksums_path=args.checksums,
            candidates_path=args.candidates,
        )
        _write(args.report_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        _write(args.report_md, markdown_report(report))
    except (InputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status'].upper()}: wrote {args.report_json} and {args.report_md}")
    if report["status"] != "valid":
        return 1
    if args.fail_on_warnings and report["validation"]["warning_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
