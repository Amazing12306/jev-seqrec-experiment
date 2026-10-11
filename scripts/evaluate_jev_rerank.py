"""Offline Jev reranking evaluator. It never sends an API request."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol import sliced_metrics, write_json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def z(values):
    x = np.asarray(values, dtype=float)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def jev_score(record):
    probs = record.get("probabilities") or {}
    candidate_ids = record["candidate_ids"]
    values = np.asarray([max(float(probs.get(f"c{i:03d}", 0.0)), 1e-12)
                        for i in range(len(candidate_ids))])
    return values, float(values.max() - np.partition(values, -2)[-2]) if len(values) > 1 else 0.0


def rerank(base, jev, alpha, confidence_threshold=None):
    by_id = {r["case_id"]: r for r in jev}
    out, used, invalid = [], 0, 0
    for row in base:
        api = by_id.get(row["case_id"])
        base_ids = row["candidate_ids"]
        base_scores = row["candidate_scores"]
        final_ids = list(base_ids)
        confidence = None
        fallback = "missing_response"
        if api is not None:
            probs, confidence = jev_score({"candidate_ids": base_ids, **api})
            if len(probs) == len(base_ids) and np.isfinite(probs).all() and float(probs.sum()) > 0:
                if confidence_threshold is None or confidence >= confidence_threshold:
                    final = alpha * z(base_scores) + (1.0 - alpha) * z(np.log(probs))
                    final_ids = [x for _, x in sorted(zip(final, base_ids), reverse=True)]
                    used += 1
                    fallback = "used"
                else:
                    fallback = "low_confidence"
            else:
                invalid += 1
                fallback = "invalid_response"
        target = int(row["target_id"])
        rank = final_ids.index(target) + 1 if target and target in final_ids else 0
        enriched = dict(row)
        enriched.update({"rank": rank, "rank_candidate": rank, "rerank_confidence": confidence,
                         "fallback_reason": fallback, "rerank_used": fallback == "used"})
        out.append(enriched)
    return out, {"n": len(base), "used": used, "invalid": invalid,
                 "coverage": used / len(base) if base else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--confidence-threshold", type=float, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = read_jsonl(root / args.candidates)
    jev = read_jsonl(root / args.jev)
    records, status = rerank(base, jev, args.alpha, args.confidence_threshold)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.with_suffix(".jsonl")).open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out.with_suffix(".metrics.json"), {"alpha": args.alpha,
        "confidence_threshold": args.confidence_threshold, "status": status,
        "metrics": sliced_metrics(records)})
    print(json.dumps({"out": str(out), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
