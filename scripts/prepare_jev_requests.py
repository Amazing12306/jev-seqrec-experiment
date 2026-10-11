"""Create auditable, TypeSafe-compatible Jev choice requests.

This command never calls the API.  It builds the exact ``state/model/questions``
body used by the caller and keeps evaluation-only fields in ``audit``.  The
retriever score is intentionally not sent to Jev: it is retained for the
downstream score fusion only.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml1m")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-method", default="SASRec-dev")
    ap.add_argument("--state-variant", choices=("full", "recent_only", "long_only", "no_gap", "no_conflict"),
                    default="full",
                    help="Ablate Jev inputs while keeping the candidate pool and case IDs fixed")
    ap.add_argument("--rich-metadata", action="store_true",
                    help="Include item description/features when available")
    ap.add_argument("--metadata-char-limit", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / args.data
    item_rows = json.loads((data / "items.json").read_text(encoding="utf-8"))
    items = {int(x["item_id"]): x for x in item_rows}
    items_by_movie_id = {int(x["movie_id"]): x for x in item_rows}
    stats = json.loads((data / "dataset_stats.json").read_text(encoding="utf-8"))
    genre_names = stats.get("genre_names", stats.get("category_names"))
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for case in read_jsonl(root / args.candidates):
            if args.limit is not None and count >= args.limit:
                break
            choices = {}
            for idx, (item_id, base_score) in enumerate(zip(case["candidate_ids"], case["candidate_scores"])):
                meta = items.get(int(item_id))
                if meta is None:
                    raise ValueError(f"candidate item {item_id} has no pre-cutoff metadata")
                choices[f"c{idx:03d}"] = {
                    "item_id": int(item_id), "title": meta["title"], "genres": meta["genres"],
                    "base_rank": idx + 1, "base_score": float(base_score),
                }
            state = case["state"]
            def named_distribution(values):
                return [{"genre": genre_names[i], "weight": round(float(v), 6)}
                        for i, v in enumerate(values) if float(v) > 0]

            # Keep the state human-readable while retaining enough structure for
            # an audit.  MovieLens has ratings only, so behavior_type is explicit.
            # ``recent_movie_ids`` uses original MovieLens movie_id values;
            # candidates use contiguous train item_id values.
            recent_items = [
                {"movie_id": int(movie_id), "title": items_by_movie_id[int(movie_id)]["title"],
                 "genres": items_by_movie_id[int(movie_id)]["genres"]}
                for movie_id in state["recent_movie_ids"]
                if int(movie_id) in items_by_movie_id
            ]
            state_payload = {
                "task": "next positive interaction in a sequence recommender",
                "long_term_preferences": named_distribution(state["long_distribution"]),
                "recent_session_intent": named_distribution(state["recent_distribution"]),
                "recent_items": recent_items,
                "time_gaps_seconds": state["observed_time_gaps_seconds"],
                "current_session_length": int(state["current_session_length"]),
                "behavior_type": state["behavior_type"],
                "conflict": {
                    "strength": state["conflict_strength"],
                    "jsd_bits": state["jsd_bits"],
                    "long_top_genre": state["long_top_genre"],
                    "recent_top_genre": state["recent_top_genre"],
                },
            }
            if args.state_variant == "recent_only":
                state_payload.pop("long_term_preferences")
                state_payload.pop("conflict")
                state_payload["decision_scope"] = "Use the recent session and time gaps only."
            elif args.state_variant == "long_only":
                state_payload.pop("recent_session_intent")
                state_payload.pop("recent_items")
                state_payload.pop("time_gaps_seconds")
                state_payload.pop("current_session_length")
                state_payload.pop("conflict")
                state_payload["decision_scope"] = "Use the long-term preference only; recent-session and time-gap features are unavailable."
            elif args.state_variant == "no_gap":
                state_payload.pop("time_gaps_seconds")
                state_payload["decision_scope"] = "Use long-term preference and recent session; time gaps are unavailable."
            elif args.state_variant == "no_conflict":
                state_payload.pop("conflict")
                state_payload["decision_scope"] = "Infer the balance between long-term preference and recent session without a conflict label."
            variant_instructions = {
                "full": "Balance long-term preference with the current session intent and the observed time gaps.",
                "recent_only": "Use only the recent session intent and observed time gaps.",
                "long_only": "Use only the long-term preference.",
                "no_gap": "Balance long-term preference and recent session intent; time gaps are unavailable.",
                "no_conflict": "Balance long-term preference, recent session intent, and time gaps without relying on a conflict label.",
            }
            instruction = ("Choose the candidate most likely to be the user's next positive "
                           "interaction. " + variant_instructions[args.state_variant] +
                           " Return probabilities for every option.")
            criteria = {}
            for key, value in choices.items():
                description = ""
                if args.rich_metadata:
                    item_meta = items[int(value["item_id"])]
                    desc = item_meta.get("description") or []
                    features = item_meta.get("features") or []
                    rich = " ".join([str(x) for x in desc[:2] + features[:3]])
                    description = f"; description/features: {rich[:args.metadata_char_limit]}" if rich else ""
                criteria[key] = (f"Item title: {value['title']}; categories: {', '.join(value['genres'])}"
                                 + description)
            request = {
                "case_id": case["case_id"],
                "model": os.environ.get("JEV_MODEL", "jev-latest"),
                "state": state_payload,
                "questions": {
                    "next_item": {
                        "type": "choice",
                        "instructions": (
                            instruction),
                        "criteria": criteria,
                    }
                },
                "audit": {
                    "target_id": int(case["target_id"]),
                    "target_known": bool(case["target_known"]),
                    "candidate_count": len(choices),
                    "base_method": args.base_method,
                    "candidate_source": args.candidates,
                    "base_scores": [float(x) for x in case["candidate_scores"]],
                    "state_source": "MovieLens rating-session proxy; no target fields included",
                    "state_variant": args.state_variant,
                    "rich_metadata": args.rich_metadata,
                    "api_called": False,
                },
            }
            if len(choices) < 2 or len(choices) > 255:
                raise ValueError(f"invalid choice count {len(choices)}")
            f.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"out": str(out), "requests": count, "api_called": False}))


if __name__ == "__main__":
    main()
