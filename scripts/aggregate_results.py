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


def user_level_stats(rows_by_seed, seeds, predicate, rng, bootstrap, permutations):
    """Average repeated seed observations within user before inference."""
    by_user = {}
    for seed in seeds:
        for baseline, reranked in rows_by_seed[seed]:
            user_id = baseline.get("user_id")
            if user_id is None or user_id != reranked.get("user_id"):
                raise ValueError(f"missing or mismatched user_id in seed {seed}")
            if predicate(baseline):
                by_user.setdefault(user_id, {"base": [], "new": []})
                by_user[user_id]["base"].append(
                    ndcg10(baseline.get("rank_candidate", baseline.get("rank", 0))))
                by_user[user_id]["new"].append(
                    ndcg10(reranked.get("rank_candidate", reranked.get("rank", 0))))

    users = sorted(by_user)
    user_base = [float(np.mean(by_user[u]["base"])) for u in users]
    user_new = [float(np.mean(by_user[u]["new"])) for u in users]
    result = paired_stats(user_base, user_new, rng, bootstrap, permutations)
    result["unit"] = "user; average over eligible seeds within user"
    result["case_observations"] = sum(len(by_user[u]["base"]) for u in users)
    return result


def conflict_advantage(rows_by_seed, seeds, rng, bootstrap, permutations):
    """Compare within-user gains on conflict vs non-conflict cases."""
    by_user = {}
    for seed in seeds:
        for baseline, reranked in rows_by_seed[seed]:
            user_id = baseline["user_id"]
            base = ndcg10(baseline.get("rank_candidate", baseline.get("rank", 0)))
            new = ndcg10(reranked.get("rank_candidate", reranked.get("rank", 0)))
            key = "conflict" if baseline.get("conflict") else (
                "nonconflict" if baseline.get("conflict_strength") == "none" else None)
            if key:
                by_user.setdefault(user_id, {"conflict": [], "nonconflict": []})[key].append(new - base)

    users = [u for u, values in by_user.items()
             if values["conflict"] and values["nonconflict"]]
    nonconflict_gain = [float(np.mean(by_user[u]["nonconflict"])) for u in users]
    conflict_gain = [float(np.mean(by_user[u]["conflict"])) for u in users]
    result = paired_stats(nonconflict_gain, conflict_gain, rng, bootstrap, permutations)
    result["unit"] = "user with both conflict and non-conflict cases"
    result["contrast"] = "conflict gain minus non-conflict gain"
    return result


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
    baselines, reranked, rows_by_seed = {}, {}, {}
    per_seed = {}
    pooled_base, pooled_new = [], []
    user_sets = {}
    for seed in seeds:
        bp = ROOT / args.baseline_template.format(seed=seed)
        rp = ROOT / args.reranked_template.format(seed=seed)
        b, r = read_jsonl(bp), read_jsonl(rp)
        by_b, by_r = {x["case_id"]: x for x in b}, {x["case_id"]: x for x in r}
        if set(by_b) != set(by_r):
            raise ValueError(f"case_id coverage mismatch for seed {seed}")
        ids = sorted(by_b)
        b, r = [by_b[i] for i in ids], [by_r[i] for i in ids]
        if any(x.get("user_id") is None for x in b):
            raise ValueError(f"missing user_id for seed {seed}")
        if len({x.get("user_id") for x in b}) != len(b):
            raise ValueError(f"expected one test case per user for seed {seed}")
        user_sets[seed] = {x.get("user_id") for x in b}
        baselines[seed], reranked[seed] = b, r
        rows_by_seed[seed] = list(zip(b, r))
        pooled_base.extend(b); pooled_new.extend(r)
        per_seed[seed] = {"baseline": metrics_from_rows(b), "reranked": metrics_from_rows(r),
                          "n": len(ids), "n_users": len(user_sets[seed])}

    if any(user_sets[seed] != user_sets[seeds[0]] for seed in seeds[1:]):
        raise ValueError("seeds do not contain the same user set; clustered inference is invalid")

    rng = np.random.default_rng(2027)
    slices = {
        "all": lambda b: True,
        "conflict": lambda b: bool(b.get("conflict")),
        "strong_conflict": lambda b: b.get("conflict_strength") == "strong",
        "eligible_nonconflict": lambda b: b.get("conflict_strength") == "none",
    }
    pooled_paired = {}
    user_level_paired = {}
    for name, pred in slices.items():
        pairs = [(b, r) for b, r in zip(pooled_base, pooled_new) if pred(b)]
        pooled_paired[name] = paired_stats(
            [ndcg10(b.get("rank_candidate", b.get("rank", 0))) for b, _ in pairs],
            [ndcg10(r.get("rank_candidate", r.get("rank", 0))) for _, r in pairs],
            rng, args.bootstrap, args.permutations)
        user_level_paired[name] = user_level_stats(
            rows_by_seed, seeds, pred, rng, args.bootstrap, args.permutations)
    user_level_paired["conflict_advantage_over_nonconflict"] = conflict_advantage(
        rows_by_seed, seeds, rng, args.bootstrap, args.permutations)
    aggregate = {}
    for key in ("ndcg@10", "conflict_ndcg@10", "strong_conflict_ndcg@10"):
        aggregate[key] = {
            "baseline": summarize_seed_values([x["baseline"][key] for x in per_seed.values()]),
            "reranked": summarize_seed_values([x["reranked"][key] for x in per_seed.values()]),
        }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "per_seed": per_seed,
                               "same_user_set_across_seeds": True,
                               "aggregate": aggregate,
                               "case_level_pooled_unadjusted": pooled_paired,
                               "user_level_paired": user_level_paired,
                               "metric": "NDCG@10", "selection_frozen_before_test": True},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "aggregate": aggregate,
                      "user_level_paired": user_level_paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
