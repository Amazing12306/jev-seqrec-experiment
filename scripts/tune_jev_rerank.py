"""Tune Jev/base fusion on a validation cache only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_jev_rerank import read_jsonl, rerank  # noqa: E402
from protocol import sliced_metrics  # noqa: E402


def parse_float_grid(text, allow_none=False):
    values = []
    for token in text.split(","):
        token = token.strip().lower()
        if allow_none and token in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--thresholds", default="none,0.2,0.4,0.6,0.8")
    args = ap.parse_args()

    base = read_jsonl(ROOT / args.candidates)
    jev = read_jsonl(ROOT / args.jev)
    results = []
    for alpha in parse_float_grid(args.alphas):
        for threshold in parse_float_grid(args.thresholds, allow_none=True):
            records, status = rerank(base, jev, alpha, threshold)
            metrics = sliced_metrics(records)
            results.append({
                "alpha": alpha,
                "confidence_threshold": threshold,
                "status": status,
                "metrics": metrics,
            })

    results.sort(key=lambda x: (
        -(x["metrics"].get("conflict", {}).get("ndcg@10") or -1),
        -(x["metrics"].get("all", {}).get("ndcg@10") or -1),
        -x["status"].get("coverage", 0.0),
    ))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "candidates": args.candidates,
        "jev": args.jev,
        "selection": "conflict.ndcg@10, then all.ndcg@10, then coverage",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "best": results[0] if results else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
