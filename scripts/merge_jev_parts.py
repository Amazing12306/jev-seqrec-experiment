"""Merge resumable Jev JSONL shards and produce an integrity manifest."""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path


def read_rows(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source"] = str(path)
                row["_line"] = line_no
                yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--parts-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected", type=int, required=True)
    args = ap.parse_args()

    paths = [args.pilot] + sorted(glob.glob(args.parts_glob))
    if not paths:
        raise SystemExit("no Jev response files found")

    # Prefer a successful response. If a case has multiple successful rows,
    # keep the last one so a later retry can replace an earlier response.
    rows_by_case = {}
    total_rows = 0
    duplicate_rows = 0
    error_types = collections.Counter()
    for row in read_rows(paths):
        total_rows += 1
        case_id = row.get("case_id")
        if not case_id:
            error_types["missing_case_id"] += 1
            continue
        if case_id in rows_by_case:
            duplicate_rows += 1
        previous = rows_by_case.get(case_id)
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            rows_by_case[case_id] = row
        if row.get("error"):
            error_types[str(row["error"])[:200]] += 1

    merged = []
    failed = []
    for case_id, row in sorted(rows_by_case.items()):
        row = dict(row)
        row.pop("_source", None)
        row.pop("_line", None)
        if row.get("error") is None:
            merged.append(row)
        else:
            failed.append(case_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    all_case_ids = set(rows_by_case)
    expected_ids = {f"valid_{i}" for i in range(1, args.expected + 1)}
    manifest = {
        "input_files": paths,
        "expected_cases": args.expected,
        "total_input_rows": total_rows,
        "unique_cases": len(all_case_ids),
        "successful_cases": len(merged),
        "failed_cases": len(failed),
        "duplicate_rows": duplicate_rows,
        "missing_expected_cases": sorted(expected_ids - all_case_ids),
        "failed_case_ids": failed,
        "error_types": dict(error_types),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "total_input_rows", "unique_cases", "successful_cases", "failed_cases",
        "duplicate_rows", "missing_expected_cases")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
