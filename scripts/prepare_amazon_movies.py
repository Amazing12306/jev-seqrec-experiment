"""Prepare Amazon Reviews'23 Movies_and_TV timestamp-with-history files.

The output follows the project JSONL contract while retaining item text fields
for the Jev reranker.  This script is intentionally sampleable because the
full 5-core category has hundreds of thousands of users.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def read_meta(path, wanted):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            asin = row.get("parent_asin") or row.get("asin")
            if asin in wanted:
                out[asin] = row
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_list(value):
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def state_for(history, items, category_names, session_gap_days=30, short_window=5):
    if not history:
        raise ValueError("empty history")
    times = [int(x[1]) for x in history]
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    gap_seconds = 86400 * session_gap_days
    breaks = [i + 1 for i, gap in enumerate(gaps) if gap >= gap_seconds]
    start = breaks[-1] if breaks else 0
    short = history[max(start, len(history) - short_window):]
    now = times[-1]
    long = [e for e in history[:start] if now - e[1] <= 365 * 86400]

    def distribution(events, decay):
        values = [0.0] * len(category_names)
        for item_id, timestamp, _ in events:
            cats = items[item_id]["genres"]
            weight = math.exp(-(now - timestamp) / (90 * 86400)) if decay else 1.0
            for cat in cats:
                if cat in category_names:
                    values[category_names.index(cat)] += weight / max(1, len(cats))
        total = sum(values)
        return [x / total for x in values] if total else values

    long_p, recent_p = distribution(long, True), distribution(short, False)
    jsd = None
    if sum(long_p) and sum(recent_p):
        m = [(a + b) / 2 for a, b in zip(long_p, recent_p)]
        jsd = 0.5 * sum(a * math.log2(a / b) for a, b in zip(long_p, m) if a > 0)
        jsd += 0.5 * sum(a * math.log2(a / b) for a, b in zip(recent_p, m) if a > 0)
    eligible = len(long) >= 2 and len(short) >= 2 and jsd is not None
    mismatch = bool(eligible and max(range(len(long_p)), key=long_p.__getitem__) !=
                    max(range(len(recent_p)), key=recent_p.__getitem__))
    strength = "ineligible"
    if eligible:
        strength = "strong" if mismatch and jsd >= 0.5 else "medium" if mismatch and jsd >= 0.2 else "none"
    return {
        "state_cutoff": now, "long_count": len(long), "recent_count": len(short),
        "current_session_length": len(history) - start,
        "long_distribution": long_p, "recent_distribution": recent_p,
        "recent_movie_ids": [int(e[0]) for e in short],
        "observed_time_gaps_seconds": gaps[-short_window + 1:], "jsd_bits": jsd,
        "top_category_mismatch": mismatch, "conflict_eligible": eligible,
        "conflict_strength": strength, "conflict": strength in ("medium", "strong"),
        "long_top_genre": category_names[max(range(len(long_p)), key=long_p.__getitem__)] if long_p and sum(long_p) else None,
        "recent_top_genre": category_names[max(range(len(recent_p)), key=recent_p.__getitem__)],
        "behavior_type": "rating",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/amazon_raw")
    ap.add_argument("--out", default="data/amazon_movies")
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw, out = root / args.raw, root / args.out
    out.mkdir(parents=True, exist_ok=True)
    train_path = raw / "Movies_and_TV.train.csv.gz"
    valid_path = raw / "Movies_and_TV.valid.csv.gz"
    test_path = raw / "Movies_and_TV.test.csv.gz"
    meta_path = raw / "meta_Movies_and_TV.jsonl.gz"
    events = defaultdict(list)
    for row in read_rows(train_path):
        events[row["user_id"]].append((row["parent_asin"], int(row["timestamp"]), float(row["rating"])))
    valid_raw = list(read_rows(valid_path))
    test_raw = list(read_rows(test_path))
    eligible_users = sorted(set(events) & {r["user_id"] for r in valid_raw} & {r["user_id"] for r in test_raw})
    rng = random.Random(args.seed)
    if args.max_users and len(eligible_users) > args.max_users:
        eligible_users = sorted(rng.sample(eligible_users, args.max_users))
    keep_users = set(eligible_users)
    item_freq = Counter(asin for user in keep_users for asin, _, _ in events[user])
    keep_asins = {asin for asin, count in item_freq.items() if count >= 5}
    meta = read_meta(meta_path, keep_asins)
    keep_asins &= set(meta)
    item_ids = {asin: i + 1 for i, asin in enumerate(sorted(keep_asins))}
    cats = Counter()
    for asin in keep_asins:
        cats.update(clean_list(meta[asin].get("categories")))
    category_names = [x for x, _ in cats.most_common(50)] or ["Unknown"]
    items = []
    for asin, item_id in sorted(item_ids.items(), key=lambda x: x[1]):
        row = meta[asin]
        categories = clean_list(row.get("categories")) or ["Unknown"]
        description = clean_list(row.get("description"))
        features = clean_list(row.get("features"))
        items.append({"item_id": item_id, "movie_id": item_id, "asin": asin,
                      "title": str(row.get("title") or asin), "genres": categories,
                      "categories": categories, "description": description,
                      "features": features, "train_frequency": item_freq[asin]})
    by_id = {x["item_id"]: x for x in items}
    by_asin = {x["asin"]: x for x in items}

    def make_row(row, split):
        history_asins = row.get("history", "").split()
        history = [by_asin[a]["item_id"] for a in history_asins if a in by_asin]
        target = by_asin.get(row["parent_asin"])
        if not history:
            return None
        prefix = [(item, int(i), 0.0) for item, i in
                  zip(history, [int(row["timestamp"]) - (len(history) - j) * 86400 for j in range(len(history))])]
        state = state_for(prefix, by_id, category_names)
        return {"case_id": f"{split}_{row['user_id']}", "user_id": row["user_id"], "split": split,
                "history": history, "target_id": target["item_id"] if target else 0,
                "target_movie_id": target["item_id"] if target else 0,
                "target_known": bool(target), "target_train_frequency": target["train_frequency"] if target else 0,
                "train_length": len(events[row["user_id"]]), "target_timestamp_group_size": 1,
                "state": state}

    train_rows = [{"user_id": u, "items": [by_asin[a]["item_id"] for a, _, _ in sorted(events[u], key=lambda x: (x[1], x[0])) if a in by_asin]}
                  for u in eligible_users]
    eval_rows = {}
    for name, raw_rows in (("valid", valid_raw), ("test", test_raw)):
        rows = [x for r in raw_rows if r["user_id"] in keep_users and r["user_id"] in events
                for x in [make_row(r, name)] if x is not None]
        eval_rows[name] = rows
    for name, rows in [("train", train_rows), *eval_rows.items()]:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    stats = {"dataset": "Amazon Reviews'23 Movies_and_TV 5-core timestamp_w_his",
             "users": len(train_rows), "items": len(items), "category_names": category_names,
             "valid": len(eval_rows["valid"]), "test": len(eval_rows["test"]),
             "max_users": args.max_users, "raw_hashes": {p.name: sha256(p) for p in (train_path, valid_path, test_path, meta_path)}}
    (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
