"""Diagnostic long/recent state scoring on a frozen candidate pool.

These are probes for the conflict construction, not claimed learned baselines.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol import sliced_metrics, write_json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def item_vector(meta, item_id, dim):
    v = np.zeros(dim, dtype=float)
    for g in meta[int(item_id)]["genres"]:
        v[meta["genres"].index(g)] += 1.0 / max(1, len(meta[int(item_id)]["genres"]))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / args.data
    items = json.loads((data / "items.json").read_text(encoding="utf-8"))
    genres = sorted({g for x in items for g in x["genres"]})
    meta = {int(x["item_id"]): x for x in items}
    meta["genres"] = genres
    cases = read_jsonl(root / args.candidates)
    all_results = {}
    for method in ("recent", "long", "equal", "adaptive"):
        records = []
        for row in cases:
            state = row["state"]
            long = np.asarray(state["long_distribution"], dtype=float)
            recent = np.asarray(state["recent_distribution"], dtype=float)
            if method == "recent":
                weights = recent
            elif method == "long":
                weights = long
            elif method == "equal":
                weights = (long + recent) / 2.0
            else:
                jsd = float(state["jsd_bits"] or 0.0)
                # Diagnostic gate: when long/recent disagree, give the recent
                # session more weight; this is not a learned gating model.
                recent_weight = min(0.8, 0.5 + jsd)
                weights = (1.0 - recent_weight) * long + recent_weight * recent
            scores = [float(item_vector(meta, item_id, len(genres)) @ weights)
                      for item_id in row["candidate_ids"]]
            order = [x for _, x in sorted(zip(scores, row["candidate_ids"]), reverse=True)]
            target = int(row["target_id"])
            rank = order.index(target) + 1 if target and target in order else 0
            records.append({"case_id": row["case_id"], "user_id": row["user_id"], "split": row["split"],
                            "target_id": target, "target_known": row["target_known"], "rank": rank,
                            "rank_candidate": rank, "candidate_ids": row["candidate_ids"],
                            "candidate_scores": scores, "conflict": row["conflict"],
                            "conflict_strength": row["conflict_strength"],
                            "target_timestamp_group_size": row["target_timestamp_group_size"],
                            "target_train_frequency": row["target_train_frequency"],
                            "train_length": row["train_length"], "state": state})
        out_path = root / args.out / f"{method}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        all_results[method] = sliced_metrics(records)
    write_json(root / args.out / "metrics.json", all_results)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
