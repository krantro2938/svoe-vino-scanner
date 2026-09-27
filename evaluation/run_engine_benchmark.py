#!/usr/bin/env python3
"""Run the in-process recognition engine over a labelled queries.jsonl.

Reports exact-slug Accuracy@1, Recall@5, per-method accuracy and latency. The
engine is created from the same environment variables as the API, so the result
describes the configuration that the service would run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))

from app.config import Settings  # noqa: E402
from app.imaging import decode_image  # noqa: E402
from app.retrieval import RecognitionEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    started = time.perf_counter()
    engine = RecognitionEngine(Settings.from_env())
    print(f"engine loaded in {time.perf_counter() - started:.1f}s: {engine.model_version}", flush=True)
    base = args.queries.parent
    results = []
    for index, row in enumerate(rows):
        payload = (base / row["image_path"]).read_bytes()
        began = time.perf_counter()
        decoded = decode_image(payload, max_pixels=engine.settings.max_image_pixels, min_side=32)
        match = engine.predict(decoded, None)
        latency = (time.perf_counter() - began) * 1000
        top5 = [slug for slug, _ in match.candidates][:5] or [match.wine.slug]
        results.append({
            "query_id": row["query_id"], "image_path": row["image_path"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "expected": row["expected_slug"], "predicted": match.wine.slug,
            "top5": top5, "method": match.method, "confidence": match.confidence,
            "margin": match.margin, "latency_ms": round(latency, 1),
        })
        if (index + 1) % 50 == 0:
            done = results
            acc = sum(r["predicted"] == r["expected"] for r in done) / len(done)
            print(f"{index + 1}/{len(rows)} acc@1={acc:.3f}", flush=True)

    correct = [r["predicted"] == r["expected"] for r in results]
    by_method: dict[str, list[bool]] = defaultdict(list)
    for r, ok in zip(results, correct):
        by_method[r["method"]].append(ok)
    latencies = sorted(r["latency_ms"] for r in results)
    summary = {
        "count": len(results),
        "accuracy_at_1": round(sum(correct) / len(results), 4),
        "recall_at_5": round(sum(r["expected"] in r["top5"] for r in results) / len(results), 4),
        "methods": {m: {"count": len(v), "accuracy": round(sum(v) / len(v), 4)} for m, v in by_method.items()},
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
        "model_version": engine.model_version,
    }
    # Calibration view: accuracy of the most confident half versus the rest.
    ordered = sorted(zip(results, correct), key=lambda pair: -pair[0]["confidence"])
    half = len(ordered) // 2
    if half:
        summary["accuracy_top_half_confidence"] = round(sum(ok for _, ok in ordered[:half]) / half, 4)
        summary["accuracy_bottom_half_confidence"] = round(
            sum(ok for _, ok in ordered[half:]) / (len(ordered) - half), 4)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Inputs for evaluation/evaluate.py (micro/macro-F1@1, Recall@5, MRR@5).
        stem = args.output.with_suffix("")
        with open(f"{stem}.predictions.jsonl", "w") as handle:
            for r in results:
                handle.write(json.dumps({"query_id": r["query_id"], "image_path": r["image_path"],
                                         "image_sha256": r["image_sha256"], "predicted_slug": r["predicted"],
                                         "latency_ms": round(r["latency_ms"])}) + "\n")
        with open(f"{stem}.candidates.jsonl", "w") as handle:
            for r in results:
                handle.write(json.dumps({"query_id": r["query_id"], "top5": r["top5"]}) + "\n")
        args.output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    confusions = Counter((r["expected"], r["predicted"]) for r, ok in zip(results, correct) if not ok)
    for (expected, predicted), n in confusions.most_common(8):
        print(f"  {expected} -> {predicted} ({n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
