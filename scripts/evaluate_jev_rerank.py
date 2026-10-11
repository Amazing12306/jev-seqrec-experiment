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
    expected = {f"c{i:03d}" for i in range(len(candidate_ids))}
    if record.get("error") or set(probs) != expected:
        raise ValueError("failed or incomplete response")
    values = np.asarray([float(probs[f"c{i:03d}"]) for i in range(len(candidate_ids))])
    if (not np.isfinite(values).all() or (values < 0).any() or
            (values > 1).any() or abs(float(values.sum()) - 1.0) > 0.02 + 1e-9):
        raise ValueError("invalid probabilities")
    values = values / values.sum()
    raw_confidence = record.get("confidence")
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
        confidence = None
    return values, confidence


def rerank(base, jev, alpha, confidence_threshold=None, tie_policy="base_rank"):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    by_id = {r["case_id"]: r for r in jev}
    if len(by_id) != len(jev):
        raise ValueError("duplicate Jev case IDs; merge the cache first")
    out, used, invalid = [], 0, 0
    for row in base:
        api = by_id.get(row["case_id"])
        base_ids = row["candidate_ids"]
        base_scores = row["candidate_scores"]
        final_ids = list(base_ids)
        confidence = None
        fallback = "missing_response"
        if api is not None:
            try:
                probs, confidence = jev_score({**api, "candidate_ids": base_ids})
            except (ValueError, TypeError):
                probs = None
            if probs is not None:
                confidence_ok = (confidence_threshold is None or
                                 (confidence is not None and confidence >= confidence_threshold))
                if confidence_ok:
                    final = alpha * z(base_scores) + (1.0 - alpha) * z(np.log(np.maximum(probs, 1e-12)))
                    if tie_policy == "item_id_legacy":
                        final_ids = [x for _, x in sorted(zip(final, base_ids), reverse=True)]
                    else:
                        final_ids = [base_ids[i] for i in np.argsort(-final, kind="stable")]
                    used += 1
                    fallback = "used"
                else:
                    fallback = "missing_confidence" if confidence is None else "low_confidence"
            else:
                invalid += 1
                fallback = "invalid_response"
        target = int(row["target_id"])
        rank = final_ids.index(target) + 1 if target and target in final_ids else 0
        enriched = dict(row)
        # Candidate files may retain both a global catalog rank (``rank``) and
        # the frozen retriever rank within the candidate pool
        # (``rank_candidate``).  Retrieval metrics and reranker comparisons
        # must use the latter when it is available.
        retriever_rank = row.get("rank_candidate", row.get("rank", 0))
        enriched.update({"rank": rank, "rank_candidate": rank, "rerank_confidence": confidence,
                         "retriever_rank": retriever_rank, "ranked_candidate_ids": final_ids,
                         "tie_policy": tie_policy,
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
    ap.add_argument("--tie-policy", choices=["base_rank", "item_id_legacy"], default="base_rank")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = read_jsonl(root / args.candidates)
    jev = read_jsonl(root / args.jev)
    records, status = rerank(base, jev, args.alpha, args.confidence_threshold, args.tie_policy)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.with_suffix(".jsonl")).open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out.with_suffix(".metrics.json"), {"alpha": args.alpha, "tie_policy": args.tie_policy,
        "confidence_threshold": args.confidence_threshold, "status": status,
        "metrics": sliced_metrics(records)})
    print(json.dumps({"out": str(out), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
