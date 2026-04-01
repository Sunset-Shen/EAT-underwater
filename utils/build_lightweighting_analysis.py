#!/usr/bin/env python3
"""Build 4.3.4 lightweighting analysis tables from compare/profile artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(v):
    if v is None or v == "":
        return None
    return float(v)


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_md_table(path: Path, rows: list[dict], fields: list[str], title: str):
    lines = [f"# {title}\n\n"]
    lines.append("| " + " | ".join(fields) + " |\n")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|\n")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compare-main-csv", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv"))
    p.add_argument("--results-root", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1"))
    p.add_argument("--output-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/lightweighting_analysis"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compare_rows = _read_csv(args.compare_main_csv)

    main_rows = []
    for r in compare_rows:
        dataset = r["dataset"]
        model = r["model"]
        profile_path = args.results_root / dataset / model / "profile" / "profile_metrics.json"
        p = _read_json(profile_path)
        main_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "accuracy": round(_to_float(r["accuracy"]), 6),
                "macro_f1": round(_to_float(r["macro_f1"]), 6),
                "trainable_params": int(p["trainable_params"]),
                "checkpoint_size_mb": round(_to_float(p["checkpoint_size_mb"]), 6),
                "flops": round(_to_float(p["flops"]), 3) if p.get("flops") is not None else "",
                "latency_ms_per_sample": round(_to_float(p["latency_ms_per_sample"]), 6),
                "throughput_samples_per_sec": round(_to_float(p["throughput_samples_per_sec"]), 6),
            }
        )

    main_fields = [
        "dataset",
        "model",
        "accuracy",
        "macro_f1",
        "trainable_params",
        "checkpoint_size_mb",
        "flops",
        "latency_ms_per_sample",
        "throughput_samples_per_sec",
    ]

    _write_csv(args.output_dir / "lightweighting_main_table.csv", main_rows, main_fields)
    _write_md_table(args.output_dir / "lightweighting_main_table.md", main_rows, main_fields, "lightweighting_main_table")

    # relative table: UT-EAT vs EAT-base per dataset
    by_ds = {}
    for r in main_rows:
        by_ds.setdefault(r["dataset"], {})[r["model"]] = r

    rel_rows = []
    for ds, d in sorted(by_ds.items()):
        b = d["eat_base"]
        u = d["ut_eat"]
        params_ratio = u["trainable_params"] / b["trainable_params"]
        size_ratio = u["checkpoint_size_mb"] / b["checkpoint_size_mb"]
        flops_ratio = u["flops"] / b["flops"]
        latency_ratio = u["latency_ms_per_sample"] / b["latency_ms_per_sample"]
        throughput_ratio = u["throughput_samples_per_sec"] / b["throughput_samples_per_sec"]

        rel_rows.append(
            {
                "dataset": ds,
                "params_ratio": round(params_ratio, 6),
                "size_ratio": round(size_ratio, 6),
                "flops_ratio": round(flops_ratio, 6),
                "latency_ratio": round(latency_ratio, 6),
                "throughput_ratio": round(throughput_ratio, 6),
                "accuracy_delta": round(u["accuracy"] - b["accuracy"], 6),
                "macro_f1_delta": round(u["macro_f1"] - b["macro_f1"], 6),
                "params_reduction_pct": round((1 - params_ratio) * 100, 3),
                "size_reduction_pct": round((1 - size_ratio) * 100, 3),
                "flops_reduction_pct": round((1 - flops_ratio) * 100, 3),
                "latency_reduction_pct": round((1 - latency_ratio) * 100, 3),
                "throughput_gain_x": round(throughput_ratio, 6),
            }
        )

    rel_fields = [
        "dataset",
        "params_ratio",
        "size_ratio",
        "flops_ratio",
        "latency_ratio",
        "throughput_ratio",
        "accuracy_delta",
        "macro_f1_delta",
        "params_reduction_pct",
        "size_reduction_pct",
        "flops_reduction_pct",
        "latency_reduction_pct",
        "throughput_gain_x",
    ]

    _write_csv(args.output_dir / "lightweighting_relative_table.csv", rel_rows, rel_fields)
    _write_md_table(args.output_dir / "lightweighting_relative_table.md", rel_rows, rel_fields, "lightweighting_relative_table")
    print(f"Wrote tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
