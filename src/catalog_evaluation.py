"""Batched full-catalog evaluation for trained sequence models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from protocol import sliced_metrics


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pad_numpy(histories, maxlen):
    batch = np.zeros((len(histories), maxlen), dtype=np.int64)
    for i, history in enumerate(histories):
        seq = history[-maxlen:]
        if seq:
            batch[i, -len(seq):] = seq
    return batch


def ranked_record(row, ids, scores, full_rank, exclude_seen):
    target = int(row["target_id"])
    pool_rank = ids.index(target) + 1 if target in ids else 0
    return {
        "case_id": row["case_id"], "user_id": row["user_id"], "split": row["split"],
        "target_id": target, "target_known": row["target_known"],
        "rank": full_rank, "rank_candidate": pool_rank,
        "candidate_ids": ids, "candidate_scores": scores,
        "conflict": row["state"]["conflict"],
        "conflict_strength": row["state"]["conflict_strength"],
        "target_timestamp_group_size": row["target_timestamp_group_size"],
        "target_train_frequency": row["target_train_frequency"],
        "train_length": row["train_length"], "state": row["state"],
        "exclude_seen": exclude_seen,
    }


def evaluate_catalog(model, rows, maxlen, candidate_limit=200, batch_size=64,
                     exclude_seen=False):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            scores = model.score_catalog(pad_numpy([r["history"] for r in batch], maxlen))
            scores = scores.detach().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite model scores")
            for row, values in zip(batch, scores):
                values = values.copy()
                if exclude_seen:
                    seen = np.asarray(row["history"], dtype=int)
                    values[seen[seen > 0] - 1] = -np.inf
                order = np.argsort(-values, kind="stable")
                order = order[np.isfinite(values[order])]
                target = int(row["target_id"])
                matches = np.flatnonzero(order == target - 1) if target else []
                full_rank = int(matches[0]) + 1 if len(matches) else 0
                pool = order[:candidate_limit]
                records.append(ranked_record(row, (pool + 1).tolist(),
                                             values[pool].tolist(), full_rank, exclude_seen))
    return records


def write_records(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def result_metrics(records):
    result = sliced_metrics(records)
    result["candidate_recall@200_full"] = float(np.mean([r["rank_candidate"] > 0 for r in records]))
    return result
