#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("shipsear", "eat_base"),
        ("shipsear", "ut_eat"),
        ("deepship", "eat_base"),
        ("deepship", "ut_eat"),
    ]

    rows = []
    for dataset, model_dir in pairs:
        eval_path = args.results_root / dataset / model_dir / "eval" / "eval_metrics.json"
        profile_path = args.results_root / dataset / model_dir / "profile" / "profile_metrics.json"
        e = _load_json(eval_path) or {}
        p = _load_json(profile_path) or {}
        rows.append(
            {
                "dataset": dataset,
                "model": model_dir,
                "init_checkpoint": e.get("init_checkpoint", ""),
                "finetuned_checkpoint": e.get("finetuned_checkpoint") or p.get("finetuned_checkpoint", ""),
                "accuracy": e.get("accuracy"),
                "macro_f1": e.get("macro_f1"),
                "trainable_params": p.get("trainable_params"),
                "checkpoint_size_mb": p.get("checkpoint_size_mb"),
                "flops": p.get("flops"),
                "latency_ms_per_sample": p.get("latency_ms_per_sample"),
                "throughput_samples_per_sec": p.get("throughput_samples_per_sec"),
            }
        )

    csv_path = args.output_dir / "compare_main.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_path = args.output_dir / "compare_main.md"
    header = "| " + " | ".join(fields) + " |\n"
    sep = "|" + "|".join(["---"] * len(fields)) + "|\n"
    lines = ["# compare_v1 summary\n\n", header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in fields) + " |\n")
    md_path.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
