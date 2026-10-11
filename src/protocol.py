"""Shared data, state and metric contract. No model training or API side effects."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def padded(history, maxlen):
    out = np.zeros(maxlen, dtype=np.int64)
    seq = np.asarray([int(i) for i in history if int(i) > 0][-maxlen:], dtype=np.int64)
    if len(seq):
        out[-len(seq):] = seq
    return out


def jsd_bits(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    def kl(x):
        keep = x > 0
        return np.sum(x[keep] * np.log2(x[keep] / m[keep]))
    return float(0.5 * kl(p) + 0.5 * kl(q))


def interest_state(prefix, metadata, genre_names, cfg):
    """Only the observed prefix is accepted; target time/item are unavailable here."""
    if not prefix:
        raise ValueError("empty prefix")
    times = np.array([e[1] for e in prefix], dtype=np.int64)
    if (np.diff(times) < 0).any():
        raise ValueError("prefix must be chronological")
    # A fixed window of the last observed session; long history excludes that session.
    breaks = np.flatnonzero(np.diff(times) >= cfg["session_gap_seconds"]) + 1
    start = int(breaks[-1]) if len(breaks) else 0
    short = prefix[max(start, len(prefix) - cfg["short_window"]):]
    now = int(times[-1])
    long = [e for e in prefix[:start]
            if now - e[1] <= cfg["long_window_days"] * 86400]
    def distribution(events, decay):
        p = np.zeros(len(genre_names), dtype=float)
        for movie_id, timestamp, rating in events:
            genres = metadata[movie_id]["genres"]
            weight = math.exp(-(now - timestamp) / (cfg["long_decay_days"] * 86400)) if decay else 1.0
            for g in genres:
                p[genre_names.index(g)] += weight / max(1, len(genres))
        return p / p.sum() if p.sum() else p
    long_p, short_p = distribution(long, True), distribution(short, False)
    jsd = jsd_bits(long_p, short_p)
    eligible = len(long) >= 2 and len(short) >= 2 and jsd is not None
    top_mismatch = bool(eligible and np.argmax(long_p) != np.argmax(short_p))
    strength = "ineligible"
    if eligible:
        strength = ("strong" if top_mismatch and jsd >= cfg["conflict_strong"] else
                    "medium" if top_mismatch and jsd >= cfg["conflict_medium"] else "none")
    gaps = np.diff(times[-cfg["short_window"]:]).astype(int).tolist()
    return {
        "state_cutoff": now, "long_count": len(long), "recent_count": len(short),
        "current_session_length": len(prefix) - start,
        "long_span_days": (now - long[0][1]) / 86400 if long else 0.0,
        "long_distribution": long_p.tolist(), "recent_distribution": short_p.tolist(),
        "recent_movie_ids": [int(e[0]) for e in short],
        "observed_time_gaps_seconds": gaps, "jsd_bits": jsd,
        "top_category_mismatch": top_mismatch, "conflict_eligible": bool(eligible),
        "conflict_strength": strength,
        "conflict": strength in ("medium", "strong"),
        "long_top_genre": genre_names[int(np.argmax(long_p))] if len(long) else None,
        "recent_top_genre": genre_names[int(np.argmax(short_p))],
        "behavior_type": "positive_rating",  # no invented click/purchase signal
    }


def rank_metrics(rank, ks=(1, 5, 10, 20)):
    # rank=0 means target absent/unavailable; MRR must be zero, not 1/(N+1).
    result = {"mrr": 1.0 / rank if rank > 0 else 0.0}
    for k in ks:
        hit = float(0 < rank <= k)
        result[f"hit@{k}"] = hit
        result[f"recall@{k}"] = hit  # single-positive next-item contract
        result[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
    return result


def summarize(rows):
    if not rows:
        return {"n": 0}
    vals = [rank_metrics(int(r["rank"])) for r in rows]
    out = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    out["n"] = len(rows)
    out["warm_target_n"] = sum(int(r["target_known"]) for r in rows)
    for n in (20, 50, 100, 200):
        # Retrieval is frozen before reranking. Measure each pool's recall
        # using the original retriever rank, not the reranked target position.
        present = [r for r in rows if 0 < r.get("retriever_rank", r["rank"]) <= n]
        out[f"candidate_recall@{n}"] = len(present) / len(rows)
        out[f"conditional_ndcg@10_given_retrieved@{n}"] = (
            float(np.mean([rank_metrics(r["rank"])["ndcg@10"] for r in present])) if present else None)
    return out


def sliced_metrics(rows):
    slices = {"all": rows,
              "conflict": [r for r in rows if r["conflict"]],
              "strong_conflict": [r for r in rows if r["conflict_strength"] == "strong"],
              "eligible_nonconflict": [r for r in rows if r["conflict_strength"] == "none"],
              "no_timestamp_tie": [r for r in rows if r["target_timestamp_group_size"] == 1],
              "tail_target_le5": [r for r in rows if r["target_train_frequency"] <= 5],
              "sparse_user_le30": [r for r in rows if r["train_length"] <= 30]}
    return {name: summarize(part) for name, part in slices.items()}
