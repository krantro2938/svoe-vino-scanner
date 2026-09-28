#!/usr/bin/env python3
"""Score a fusion model on a labelled real-photo set (cached features).

Queries whose expected slug is NOT_IN_CATALOG measure the open-set behaviour:
how often the service says "not found" (p(top-1) below the threshold) and how
often it is confidently wrong about a wine that is not in the catalogue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))
from app.pipeline import FusionModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, default=ROOT / "services/inference/data/fusion.json")
    parser.add_argument("--found", type=float, default=0.80)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    model = FusionModel.load(args.fusion)
    accepted = {json.loads(l)["query_id"]: set(json.loads(l).get("accepted_slugs") or [json.loads(l)["expected_slug"]])
                for l in args.queries.read_text().splitlines()}
    known, unknown = [], []
    for line in args.features.read_text().splitlines():
        row = json.loads(line)
        cands = row["candidates"]
        p = model.probabilities([{k: v for k, v in c.items() if k not in ("slug", "label")} for c in cands])
        order = np.argsort(-p)
        top = [cands[i]["slug"] for i in order[:5]]
        exp = accepted[row["query_id"]]
        record = (row["query_id"], float(p[order[0]]), top, exp)
        (unknown if "NOT_IN_CATALOG" in exp else known).append(record)
    thr = model.not_found_threshold or 0.45
    t1 = sum(r[2][0] in r[3] for r in known)
    t5 = sum(bool(set(r[2]) & r[3]) for r in known)
    print(f"in-catalogue  n={len(known)}: top-1 {t1}/{len(known)} ({t1 / len(known):.0%}), "
          f"top-5 {t5}/{len(known)} ({t5 / len(known):.0%}), "
          f"found(p>={args.found}) {sum(r[1] >= args.found for r in known)} "
          f"of which correct {sum(r[1] >= args.found and r[2][0] in r[3] for r in known)}, "
          f"wrongly not_found {sum(r[1] < thr for r in known)}")
    if unknown:
        print(f"off-catalogue n={len(unknown)}: not_found {sum(r[1] < thr for r in unknown)} "
              f"({sum(r[1] < thr for r in unknown) / len(unknown):.0%}), "
              f"confidently wrong (p>={args.found}) {sum(r[1] >= args.found for r in unknown)}")
    if args.verbose:
        for qid, p1, top, exp in known:
            rank = next((i for i, s in enumerate(top) if s in exp), "-")
            print(f"  {'OK ' if top[0] in exp else 'BAD'} {qid} p={p1:.2f} rank={rank} got={top[0][:55]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
