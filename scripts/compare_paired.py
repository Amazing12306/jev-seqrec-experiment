"""Paired user-level comparison for baseline and Jev reranked JSONL results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ndcg10(rank):
    rank = int(rank or 0)
    return 1.0 / np.log2(rank + 1) if 0 < rank <= 10 else 0.0


def paired_stats(base_values, new_values, rng, bootstrap=2000, permutations=10000):
    base = np.asarray(base_values, dtype=float)
    new = np.asarray(new_values, dtype=float)
    diff = new - base
    n = len(diff)
    if n == 0:
        return {"n": 0}
    boot_idx = rng.integers(0, n, size=(bootstrap, n))
    boot_means = diff[boot_idx].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, n))
    null_means = (diff[None, :] * signs).mean(axis=1)
    observed = float(diff.mean())
    p_value = float((np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1) /
                    (permutations + 1))
    return {
        "n": n,
        "baseline_mean": float(base.mean()),
        "reranked_mean": float(new.mean()),
        "delta": observed,
        "bootstrap_ci95": [float(np.quantile(boot_means, 0.025)),
                            float(np.quantile(boot_means, 0.975))],
        "paired_sign_permutation_p": p_value,
        "win_rate": float(np.mean(diff > 0)),
        "tie_rate": float(np.mean(diff == 0)),
        "loss_rate": float(np.mean(diff < 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--reranked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=10000)
    args = ap.parse_args()

    base = {row["case_id"]: row for row in read_jsonl(args.baseline)}
    reranked = {row["case_id"]: row for row in read_jsonl(args.reranked)}
    case_ids = sorted(set(base) & set(reranked))
    rows = [(base[c], reranked[c]) for c in case_ids]
    slices = {
        "all": lambda row: True,
        "conflict": lambda row: bool(row[0].get("conflict")),
        "strong_conflict": lambda row: row[0].get("conflict_strength") == "strong",
        "eligible_nonconflict": lambda row: row[0].get("conflict_strength") == "none",
        "tail_target_le5": lambda row: int(row[0].get("target_train_frequency", 0)) <= 5,
        "sparse_user_le30": lambda row: int(row[0].get("train_length", 0)) <= 30,
    }
    rng = np.random.default_rng(args.seed)
    results = {}
    for name, predicate in slices.items():
        selected = [pair for pair in rows if predicate(pair)]
        base_values = [ndcg10(pair[0].get("rank_candidate", pair[0].get("rank", 0)))
                       for pair in selected]
        new_values = [ndcg10(pair[1].get("rank_candidate", pair[1].get("rank", 0)))
                      for pair in selected]
        results[name] = paired_stats(base_values, new_values, rng,
                                     args.bootstrap, args.permutations)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "baseline": args.baseline,
        "reranked": args.reranked,
        "metric": "NDCG@10",
        "unit": "user/case paired",
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "all": results["all"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
