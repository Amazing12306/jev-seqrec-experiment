"""Evaluate deterministic traditional next-item baselines.

The baselines use only the frozen training split.  They can be evaluated on
the full catalog or rescored on an existing candidate JSONL.  The latter is
useful for isolating a Jev reranker from candidate generation without making
another hosted-API call.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protocol import sliced_metrics, write_json  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_counts(users):
    popularity = Counter()
    transitions = defaultdict(Counter)
    for user in users:
        items = [int(x) for x in user["items"]]
        popularity.update(items)
        for previous, current in zip(items, items[1:]):
            transitions[previous][current] += 1
    return popularity, transitions


def score(method, history, item_ids, popularity, transitions):
    if method == "popularity":
        return [float(popularity.get(item_id, 0)) for item_id in item_ids]
    if method != "markov1":
        raise ValueError(f"unknown method: {method}")
    previous = int(history[-1]) if history else 0
    next_counts = transitions.get(previous, {})
    # Transition frequency is primary.  A tiny fixed popularity backoff gives
    # deterministic ordering among unseen transitions without tuning on test.
    return [float(next_counts.get(item_id, 0)) +
            1e-6 * float(popularity.get(item_id, 0)) for item_id in item_ids]


def ranked_order(item_ids, values):
    # Stable item-id tie breaking makes the output independent of hash order.
    return [item_id for _, item_id in sorted(
        zip(values, item_ids), key=lambda pair: (-pair[0], pair[1]))]


def make_record(row, order, values, candidate_limit=200, common_pool=False):
    target = int(row["target_id"])
    if common_pool:
        ids = order[:candidate_limit]
        pool_rank = ids.index(target) + 1 if target in ids else 0
        full_rank = pool_rank
        scores = values
    else:
        ids = order[:candidate_limit]
        pool_rank = ids.index(target) + 1 if target in ids else 0
        full_rank = order.index(target) + 1 if target in order else 0
        scores = values[:candidate_limit]
    return {
        "case_id": row["case_id"], "user_id": row["user_id"],
        "split": row["split"], "target_id": target,
        "target_known": bool(row["target_known"]), "rank": full_rank,
        "rank_candidate": pool_rank, "candidate_ids": ids,
        "candidate_scores": [float(x) for x in scores],
        "conflict": bool(row["state"]["conflict"]),
        "conflict_strength": row["state"]["conflict_strength"],
        "target_timestamp_group_size": int(row["target_timestamp_group_size"]),
        "target_train_frequency": int(row["target_train_frequency"]),
        "train_length": int(row["train_length"]), "state": row["state"],
        "baseline_method": row.get("baseline_method"),
        "candidate_source": row.get("candidate_source", "full_catalog"),
    }


def evaluate(rows, method, popularity, transitions, candidate_limit,
             common_pool, movie_to_item):
    records = []
    for row in rows:
        if common_pool:
            item_ids = [int(x) for x in row["candidate_ids"]]
            recent_movies = row["state"].get("recent_movie_ids") or []
            history = [movie_to_item.get(int(x), 0) for x in recent_movies]
            history = [x for x in history if x > 0]
            values = score(method, history, item_ids, popularity, transitions)
            order_idx = np.argsort(-np.asarray(values), kind="stable")
            order = [item_ids[i] for i in order_idx]
            ordered_values = [values[i] for i in order_idx]
            record = make_record(row, order, ordered_values, candidate_limit, True)
        else:
            item_ids = list(range(1, max(popularity.keys(), default=0) + 1))
            values = score(method, row.get("history", []), item_ids,
                           popularity, transitions)
            order = ranked_order(item_ids, values)
            ordered_values = [values[item_id - 1] for item_id in order]
            record = make_record(row, order, ordered_values, candidate_limit, False)
        record["baseline_method"] = method
        records.append(record)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="popularity,markov1")
    ap.add_argument("--candidates", default=None,
                    help="Optional existing candidate JSONL; enables common-pool scoring")
    ap.add_argument("--split", choices=["valid", "test"], default=None)
    ap.add_argument("--candidate-limit", type=int, default=200)
    args = ap.parse_args()
    data = ROOT / args.data
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    users = read_jsonl(data / "train.jsonl")
    item_rows = json.loads((data / "items.json").read_text(encoding="utf-8"))
    movie_to_item = {int(row["movie_id"]): int(row["item_id"])
                     for row in item_rows}
    popularity, transitions = fit_counts(users)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    candidate_rows = None
    common_pool = args.candidates is not None
    if common_pool:
        candidate_rows = read_jsonl(ROOT / args.candidates)
        splits = [args.split] if args.split else sorted({r["split"] for r in candidate_rows})
    else:
        splits = [args.split] if args.split else ["valid", "test"]
    all_metrics = {}
    for split in splits:
        rows = candidate_rows if common_pool else read_jsonl(data / f"{split}.jsonl")
        if common_pool:
            rows = [r for r in rows if r["split"] == split]
        for method in methods:
            records = evaluate(rows, method, popularity, transitions,
                               args.candidate_limit, common_pool, movie_to_item)
            suffix = f"_{split}" if not common_pool else f"_{split}_common_pool"
            path = out / f"{method}{suffix}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False,
                                       separators=(",", ":")) + "\n")
            all_metrics[f"{method}:{split}"] = sliced_metrics(records)
    write_json(out / "metrics.json", {
        "data": args.data, "methods": methods, "common_pool": common_pool,
        "candidates": args.candidates, "metrics": all_metrics,
    })
    print(json.dumps({k: {
        "n": v["all"]["n"], "ndcg@10": v["all"]["ndcg@10"],
        "conflict_ndcg@10": v["conflict"]["ndcg@10"],
        "strong_conflict_ndcg@10": v["strong_conflict"]["ndcg@10"],
    } for k, v in all_metrics.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
