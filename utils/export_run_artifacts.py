#!/usr/bin/env python3
"""Export train curves and run summary from fairseq json logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

JSON_RE = re.compile(r"\{.*\}")


def parse_log(path: Path) -> List[Dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = JSON_RE.search(line)
        if not m:
            continue
        payload = m.group(0)
        try:
            d = json.loads(payload)
        except Exception:
            continue
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (run_dir / "train.log")
    rows = parse_log(log_file)

    curves_jsonl = run_dir / "curves.jsonl"
    with curves_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    preferred_cols = [
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_accuracy",
        "valid_best_accuracy",
        "train_num_updates",
        "valid_num_updates",
    ]
    all_cols = sorted({k for r in rows for k in r.keys()})
    cols = [c for c in preferred_cols if c in all_cols] + [
        c for c in all_cols if c not in preferred_cols
    ]

    curves_csv = run_dir / "curves.csv"
    with curves_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    valid_rows = [r for r in rows if "valid_accuracy" in r]
    best_valid_acc = None
    best_valid_update = None
    if valid_rows:
        best_row = max(valid_rows, key=lambda x: float(x.get("valid_accuracy", 0)))
        best_valid_acc = float(best_row.get("valid_accuracy", 0))
        best_valid_update = best_row.get("valid_num_updates", None)

    summary = {
        "tag": args.tag,
        "run_dir": str(run_dir),
        "log_file": str(log_file),
        "best_checkpoint": str(run_dir / "checkpoint_best.pt"),
        "last_checkpoint": str(run_dir / "checkpoint_last.pt"),
        "best_valid_accuracy": best_valid_acc,
        "best_valid_update": best_valid_update,
        "num_log_rows": len(rows),
        "curves_jsonl": str(curves_jsonl),
        "curves_csv": str(curves_csv),
    }

    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
