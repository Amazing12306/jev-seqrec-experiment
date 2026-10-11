"""Build immutable chronological splits, prefix-only conflict state and audits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol import ROOT, interest_state, read_config, sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml1m.yaml")
    args = ap.parse_args()
    cfg = read_config(args.config)
    out = ROOT / cfg["data_dir"]
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(cfg["raw_root"]) / "ml-1m.zip"
    # Read directly from the zip. No unmanaged extraction or unsafe path operations.
    with zipfile.ZipFile(archive) as z:
        with z.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(f, sep="::", engine="python", header=None,
                                  names=["user", "movie", "rating", "timestamp"])
        with z.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(f, sep="::", engine="python", encoding="latin-1", header=None,
                                 names=["movie", "title", "genres"])
    metadata = {int(r.movie): {"title": r.title, "genres": r.genres.split("|")}
                for r in movies.itertuples()}
    genre_names = sorted({g for m in metadata.values() for g in m["genres"]})
    positives = ratings[ratings.rating >= cfg["rating_min"]].sort_values(
        ["user", "timestamp", "movie"], kind="mergesort")
    raw_splits = {}
    for user, group in positives.groupby("user", sort=True):
        events = [(int(r.movie), int(r.timestamp), int(r.rating)) for r in group.itertuples()]
        stamps = sorted({e[1] for e in events})
        if len(stamps) < 3:
            continue
        val_ts, test_ts = stamps[-2:]
        train = [e for e in events if e[1] < val_ts]
        val_group = [e for e in events if e[1] == val_ts]
        test_group = [e for e in events if e[1] == test_ts]
        test_prefix = [e for e in events if e[1] < test_ts]
        if len(train) >= cfg["min_train_user_events"]:
            # Deterministic representative of a simultaneous event group; keep tie size.
            raw_splits[int(user)] = {"train": train, "valid_prefix": train,
                                    "test_prefix": test_prefix,
                                    "valid_target": val_group[-1], "test_target": test_group[-1],
                                    "valid_group_size": len(val_group), "test_group_size": len(test_group)}
    active = set(raw_splits)
    keep_items = set()
    while True:
        counts = Counter(e[0] for u in active for e in raw_splits[u]["train"])
        new_items = {i for i, c in counts.items() if c >= cfg["min_train_item_events"]}
        new_active = {u for u in active if sum(e[0] in new_items for e in raw_splits[u]["train"])
                      >= cfg["min_train_user_events"]}
        if new_active == active and new_items == keep_items:
            break
        active, keep_items = new_active, new_items
        if not active:
            raise RuntimeError("train-only filtering removed all users")
    movie_to_idx = {m: n + 1 for n, m in enumerate(sorted(keep_items))}
    freq = Counter(e[0] for u in active for e in raw_splits[u]["train"] if e[0] in keep_items)
    items = [{"item_id": idx, "movie_id": m, **metadata[m], "train_frequency": freq[m]}
             for m, idx in movie_to_idx.items()]
    write_json(out / "items.json", items)
    train_rows, eval_rows, manifest = [], {"valid": [], "test": []}, []
    strict_audit = {"future_in_prefix": 0, "overlapping_train_holdout_timestamps": 0}
    for user in sorted(active):
        split = raw_splits[user]
        train_ids = [movie_to_idx[e[0]] for e in split["train"] if e[0] in movie_to_idx]
        train_rows.append({"user_id": user, "items": train_ids})
        for mode in ("valid", "test"):
            prefix, target = split[f"{mode}_prefix"], split[f"{mode}_target"]
            if any(e[1] >= target[1] for e in prefix):
                strict_audit["future_in_prefix"] += 1
            state = interest_state(prefix, metadata, genre_names, cfg)
            row = {"case_id": f"{mode}_{user}", "user_id": user, "split": mode,
                   "history": [movie_to_idx[e[0]] for e in prefix if e[0] in movie_to_idx],
                   "target_id": movie_to_idx.get(target[0], 0), "target_movie_id": target[0],
                   "target_timestamp": target[1], "target_timestamp_group_size": split[f"{mode}_group_size"],
                   "target_known": target[0] in movie_to_idx,
                   "target_train_frequency": freq.get(target[0], 0), "train_length": len(train_ids),
                   "state": state}
            eval_rows[mode].append(row)
            manifest.append({k: row[k] for k in ("case_id", "user_id", "split", "target_known",
                                                 "target_timestamp_group_size", "train_length")})
            manifest[-1].update({k: state[k] for k in ("long_count", "recent_count", "jsd_bits",
                "long_top_genre", "recent_top_genre", "conflict_eligible", "conflict", "conflict_strength")})
    for name, rows in [("train", train_rows), *eval_rows.items()]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(manifest).to_csv(out / "conflict_manifest.csv", index=False)
    stats = {"raw_ratings": len(ratings), "positive_ratings": len(positives), "rating_min": cfg["rating_min"],
             "users": len(active), "items": len(items), "train_interactions": sum(len(r["items"]) for r in train_rows),
             "genre_names": genre_names, "audit": strict_audit,
             "protocol": {
                 "split": "last_two_distinct_timestamp_blocks; deterministic_last_movie_id_anchor",
                 "tie_policy": "all same-time positives excluded from target prefix; singleton slice reported",
                 "catalog": "train-only user/item core; unknown targets retained as rank=0 failures",
                 "cross_user_time": "per-user split permits training events later than another user's test; global-time split pending",
                 "long_short": "long excludes current observed session; short=last5 of observed session",
                 "session_semantics": "rating-session proxy, not viewing or purchasing intent",
                 "cutoff": "latest observed prefix time, never target timestamp",
                 "behavior": "ratings only; click/purchase-type ablation unavailable on MovieLens",
                 "train_ties": "deterministic movie_id order; order within timestamp is unobserved",
             },
             "raw_archive_sha256": sha256(archive)}
    for mode, rows in eval_rows.items():
        stats[mode] = {"n": len(rows), "warm_target_n": sum(r["target_known"] for r in rows),
                       "tied_target_n": sum(r["target_timestamp_group_size"] > 1 for r in rows),
                       "eligible_n": sum(r["state"]["conflict_eligible"] for r in rows),
                       "conflict_n": sum(r["state"]["conflict"] for r in rows),
                       "strong_conflict_n": sum(r["state"]["conflict_strength"] == "strong" for r in rows)}
    if strict_audit["future_in_prefix"]:
        raise RuntimeError("future leakage detected")
    hashes = {f: sha256(out / f) for f in ["train.jsonl", "valid.jsonl", "test.jsonl", "items.json",
                                         "conflict_manifest.csv"]}
    write_json(out / "hashes.json", hashes)
    (out / "split_hash.txt").write_text(hashes["train.jsonl"] + "\n" + hashes["valid.jsonl"] + "\n" + hashes["test.jsonl"] + "\n")
    (out / "conflict_manifest_hash.txt").write_text(hashes["conflict_manifest.csv"] + "\n")
    write_json(out / "dataset_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
