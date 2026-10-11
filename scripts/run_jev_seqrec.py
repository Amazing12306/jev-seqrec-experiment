#!/usr/bin/env python3
"""Call Jev for frozen sequence-recommendation requests.

The key is read from ``JEV_API_KEY`` only.  The output is resumable JSONL and
contains model/usage/latency/error metadata, but never the credential.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def call_jev(session, payload, api_key, max_retries):
    started = time.perf_counter()
    last_error = None
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            t0 = time.perf_counter()
            response = session.post(
                os.environ.get("JEV_ENDPOINT", "https://ai.xiaohaima-edu.com/v1/systemone"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=(20, 180))
            request_ms = (time.perf_counter() - t0) * 1000.0
            if response.ok:
                data = response.json()
                answer = (data.get("answers") or {}).get("next_item") or {}
                probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
                expected = set(payload["questions"]["next_item"]["criteria"])
                if set(probs) != expected or not all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values()):
                    raise ValueError("invalid or incomplete candidate probability distribution")
                raw_probability_sum = float(sum(probs.values()))
                if raw_probability_sum <= 0 or abs(raw_probability_sum - 1.0) > 0.02:
                    raise ValueError("candidate probabilities do not sum to one")
                # The service rounds probabilities to two decimals in some
                # responses (for example, a valid distribution can sum to
                # 0.99). Normalize after validation and retain the raw sum for
                # auditability.
                if raw_probability_sum != 1.0:
                    probs = {key: value / raw_probability_sum for key, value in probs.items()}
                usage = data.get("usage") or {}
                return {
                    "probabilities": probs, "choice": answer.get("choice"),
                    "confidence": answer.get("confidence"), "model": data.get("model"),
                    "raw_probability_sum": raw_probability_sum,
                    "success_latency_ms": request_ms,
                    "end_to_end_latency_ms": (time.perf_counter() - started) * 1000.0,
                    "attempts": attempts, "error": None,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code in (400, 401, 402, 403, 422):
                break
        except Exception as exc:
            last_error = repr(exc)
        if attempt < max_retries:
            time.sleep(2.0 * (2 ** attempt))
    return {"probabilities": {}, "choice": None, "confidence": None, "model": None,
            "raw_probability_sum": None,
            "success_latency_ms": None,
            "end_to_end_latency_ms": (time.perf_counter() - started) * 1000.0,
            "attempts": attempts, "error": last_error,
            "input_tokens": None, "output_tokens": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()
    if not os.environ.get("JEV_API_KEY"):
        raise SystemExit("JEV_API_KEY is not set; no request was sent")
    model = os.environ.get("JEV_MODEL", "jev-latest")
    session = requests.Session()
    api_key = os.environ["JEV_API_KEY"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for row in records(out):
            if row.get("case_id") and not row.get("error"):
                done[row["case_id"]] = row
    n = 0
    used = 0
    with out.open("a", encoding="utf-8") as f:
        for req in records(ROOT / args.requests):
            if args.limit is not None and n >= args.limit:
                break
            case_id = req["case_id"]
            if case_id in done:
                continue
            payload = {"state": req["state"], "model": model, "questions": req["questions"]}
            labels = payload["questions"]["next_item"]["criteria"]
            answer = call_jev(session, payload, api_key, args.max_retries)
            record = {
                "case_id": case_id,
                **answer,
                "candidate_count": len(labels),
                "recorded_at": time.time(),
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            n += 1
            used += int(answer["error"] is None)
            if args.sleep:
                time.sleep(args.sleep)
    print(json.dumps({"requests_processed": n, "successful": used,
                      "out": str(out), "key_present": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
