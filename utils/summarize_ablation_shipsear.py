#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_md(path: Path, rows: list[dict], fields: list[str], title: str):
    lines = [f"# {title}\n\n", "| " + " | ".join(fields) + " |\n", "|" + "|".join(["---"]*len(fields)) + "|\n"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/ablation_shipsear"))
    p.add_argument("--compare-main-csv", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # baseline from compare_v1
    baseline_acc = baseline_f1 = None
    with args.compare_main_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "shipsear" and r["model"] == "eat_base":
                baseline_acc = float(r["accuracy"])
                baseline_f1 = float(r["macro_f1"])
                break

    variant_defs = [
        ("eat_base", "no", "no", "no", None),
        ("ut_backbone", "yes", "no", "no", Path("/hy-tmp/exp/eat/results/ablation_shipsear/eval/ut_backbone_eval.json")),
        ("ut_mask", "yes", "yes", "no", Path("/hy-tmp/exp/eat/results/ablation_shipsear/eval/ut_mask_eval.json")),
        ("ut_loss", "yes", "no", "yes", Path("/hy-tmp/exp/eat/results/ablation_shipsear/eval/ut_loss_eval.json")),
        ("ut_full", "yes", "yes", "yes", Path("/hy-tmp/exp/eat/results/ablation_shipsear/eval/ut_full_eval.json")),
    ]

    main_rows = []
    for name, b, m, l, eval_path in variant_defs:
        if name == "eat_base":
            acc, f1 = baseline_acc, baseline_f1
        else:
            d = _load_json(eval_path)
            acc, f1 = float(d["accuracy"]), float(d["macro_f1"])
        main_rows.append(
            {
                "variant": name,
                "lightweight_backbone": b,
                "physical_guided_mask": m,
                "physical_loss": l,
                "accuracy": round(acc, 6),
                "macro_f1": round(f1, 6),
            }
        )

    fields_main = ["variant", "lightweight_backbone", "physical_guided_mask", "physical_loss", "accuracy", "macro_f1"]
    _write_csv(args.output_dir / "ablation_main_table.csv", main_rows, fields_main)
    _write_md(args.output_dir / "ablation_main_table.md", main_rows, fields_main, "ablation_main_table")

    ref = next(r for r in main_rows if r["variant"] == "ut_backbone")
    full = next(r for r in main_rows if r["variant"] == "ut_full")

    delta_rows = []
    for r in main_rows:
        delta_rows.append(
            {
                "variant": r["variant"],
                "accuracy_delta_vs_ut_backbone": round(r["accuracy"] - ref["accuracy"], 6),
                "macro_f1_delta_vs_ut_backbone": round(r["macro_f1"] - ref["macro_f1"], 6),
                "accuracy_delta_vs_full": round(r["accuracy"] - full["accuracy"], 6),
                "macro_f1_delta_vs_full": round(r["macro_f1"] - full["macro_f1"], 6),
            }
        )

    fields_delta = ["variant", "accuracy_delta_vs_ut_backbone", "macro_f1_delta_vs_ut_backbone", "accuracy_delta_vs_full", "macro_f1_delta_vs_full"]
    _write_csv(args.output_dir / "ablation_delta_table.csv", delta_rows, fields_delta)
    _write_md(args.output_dir / "ablation_delta_table.md", delta_rows, fields_delta, "ablation_delta_table")

    print(f"Wrote ablation summary to {args.output_dir}")


if __name__ == "__main__":
    main()
