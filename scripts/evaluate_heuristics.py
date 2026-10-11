"""Popularity and deterministic-random sanity baselines on the same ML-1M cases."""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol import rank_metrics, read_config, sliced_metrics, write_json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def evaluate(rows, ranking, top_n=200):
    records = []
    for row in rows:
        target = int(row["target_id"])
        ids = ranking[:top_n]
        rank = ids.index(target) + 1 if target in ids else 0
        records.append({
            "case_id": row["case_id"], "user_id": row["user_id"], "split": row["split"],
            "target_id": target, "target_known": bool(row["target_known"]), "rank": rank,
            "rank_candidate": rank, "candidate_ids": ids,
            "conflict": bool(row["state"]["conflict"]),
            "conflict_strength": row["state"]["conflict_strength"],
            "target_timestamp_group_size": int(row["target_timestamp_group_size"]),
            "target_train_frequency": int(row["target_train_frequency"]),
            "train_length": int(row["train_length"]), "state": row["state"],
        })
    return records


def summary(records):
    out = sliced_metrics(records)
    out["candidate_recall@200_full"] = float(np.mean([r["rank"] > 0 for r in records]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--out", default="results/dev/heuristics")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / args.data
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    users = read_jsonl(data / "train.jsonl")
    rows = read_jsonl(data / "test.jsonl")
    count = Counter(i for user in users for i in user["items"])
    all_items = sorted(count, key=lambda i: (-count[i], i))
    random_order = all_items.copy()
    random.Random(args.seed).shuffle(random_order)
    result = {}
    for name, order in (("popularity", all_items), ("random", random_order)):
        records = evaluate(rows, order)
        with (out / f"{name}_candidates.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        result[name] = summary(records)
    write_json(out / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
