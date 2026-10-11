"""Aggregate fixed-seed baseline/Jev test results without retuning on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_paired import ndcg10, paired_stats, read_jsonl  # noqa: E402


def metrics_from_rows(rows):
    values = np.asarray([ndcg10(r.get("rank")) for r in rows], dtype=float)
    conflicts = np.asarray([ndcg10(r.get("rank")) for r in rows if r.get("conflict")], dtype=float)
    strong = np.asarray([ndcg10(r.get("rank")) for r in rows if r.get("conflict_strength") == "strong"], dtype=float)
    return {"n": len(rows), "ndcg@10": float(values.mean()),
            "conflict_ndcg@10": float(conflicts.mean()),
            "strong_conflict_ndcg@10": float(strong.mean())}


def summarize_seed_values(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "values": [float(v) for v in x]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2027,2028,2029")
    ap.add_argument("--baseline-template", required=True)
    ap.add_argument("--reranked-template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=5000)
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    baselines, reranked = {}, {}
    per_seed = {}
    pooled_base, pooled_new = [], []
    for seed in seeds:
        bp = ROOT / args.baseline_template.format(seed=seed)
        rp = ROOT / args.reranked_template.format(seed=seed)
        b, r = read_jsonl(bp), read_jsonl(rp)
        by_b, by_r = {x["case_id"]: x for x in b}, {x["case_id"]: x for x in r}
        ids = sorted(set(by_b) & set(by_r))
        b, r = [by_b[i] for i in ids], [by_r[i] for i in ids]
        baselines[seed], reranked[seed] = b, r
        pooled_base.extend(b); pooled_new.extend(r)
        per_seed[seed] = {"baseline": metrics_from_rows(b), "reranked": metrics_from_rows(r), "n": len(ids)}

    rng = np.random.default_rng(2027)
    slices = {
        "all": lambda b: True,
        "conflict": lambda b: bool(b.get("conflict")),
        "strong_conflict": lambda b: b.get("conflict_strength") == "strong",
        "eligible_nonconflict": lambda b: b.get("conflict_strength") == "none",
    }
    pooled_paired = {}
    for name, pred in slices.items():
        pairs = [(b, r) for b, r in zip(pooled_base, pooled_new) if pred(b)]
        pooled_paired[name] = paired_stats(
            [ndcg10(b.get("rank_candidate", b.get("rank", 0))) for b, _ in pairs],
            [ndcg10(r.get("rank_candidate", r.get("rank", 0))) for _, r in pairs],
            rng, args.bootstrap, args.permutations)
    aggregate = {}
    for key in ("ndcg@10", "conflict_ndcg@10", "strong_conflict_ndcg@10"):
        aggregate[key] = {
            "baseline": summarize_seed_values([x["baseline"][key] for x in per_seed.values()]),
            "reranked": summarize_seed_values([x["reranked"][key] for x in per_seed.values()]),
        }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "per_seed": per_seed,
                               "aggregate": aggregate, "pooled_paired": pooled_paired,
                               "metric": "NDCG@10", "selection_frozen_before_test": True},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "aggregate": aggregate, "pooled_paired": pooled_paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
