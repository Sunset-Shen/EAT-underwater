#!/usr/bin/env python3
"""Plot 4.3.4 lightweighting/deployment figures (matplotlib only)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _to_float(v):
    return float(v)


def _norm_plot(main_rows, out_dir: Path):
    metrics = ["trainable_params", "checkpoint_size_mb", "flops", "latency_ms_per_sample"]
    ds_list = ["shipsear", "deepship"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds in zip(axes, ds_list):
        base = next(r for r in main_rows if r["dataset"] == ds and r["model"] == "eat_base")
        ute = next(r for r in main_rows if r["dataset"] == ds and r["model"] == "ut_eat")
        vals = [
            _to_float(ute[m]) / _to_float(base[m])
            for m in metrics
        ]
        x = np.arange(len(metrics))
        ax.bar(x, vals, color="#55A868")
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["Params", "Size", "FLOPs", "Latency"], rotation=20)
        ax.set_title(ds)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("UT-EAT / EAT-base")
    fig.suptitle("Normalized Lightweighting Metrics (EAT-base=1.0)")
    _save(fig, out_dir / "lightweighting_normalized_comparison.png")


def _efficiency_plot(main_rows, out_dir: Path):
    ds_list = ["shipsear", "deepship"]
    x = np.arange(len(ds_list))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lat_base = [
        _to_float(next(r for r in main_rows if r["dataset"] == ds and r["model"] == "eat_base")["latency_ms_per_sample"]) for ds in ds_list
    ]
    lat_ute = [
        _to_float(next(r for r in main_rows if r["dataset"] == ds and r["model"] == "ut_eat")["latency_ms_per_sample"]) for ds in ds_list
    ]
    axes[0].bar(x - w / 2, lat_base, width=w, color="#4C72B0", label="EAT-base")
    axes[0].bar(x + w / 2, lat_ute, width=w, color="#55A868", label="UT-EAT")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_list)
    axes[0].set_title("Latency (ms/sample)")
    axes[0].set_ylabel("ms/sample")
    for i, v in enumerate(lat_base):
        axes[0].text(i - w / 2, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(lat_ute):
        axes[0].text(i + w / 2, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)

    thr_base = [
        _to_float(next(r for r in main_rows if r["dataset"] == ds and r["model"] == "eat_base")["throughput_samples_per_sec"]) for ds in ds_list
    ]
    thr_ute = [
        _to_float(next(r for r in main_rows if r["dataset"] == ds and r["model"] == "ut_eat")["throughput_samples_per_sec"]) for ds in ds_list
    ]
    axes[1].bar(x - w / 2, thr_base, width=w, color="#4C72B0", label="EAT-base")
    axes[1].bar(x + w / 2, thr_ute, width=w, color="#55A868", label="UT-EAT")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ds_list)
    axes[1].set_title("Throughput (samples/s)")
    axes[1].set_ylabel("samples/s")
    for i, v in enumerate(thr_base):
        axes[1].text(i - w / 2, v + 4, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(thr_ute):
        axes[1].text(i + w / 2, v + 4, f"{v:.1f}", ha="center", fontsize=8)

    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Inference Efficiency Comparison")
    _save(fig, out_dir / "inference_efficiency_comparison.png")


def _tradeoff_plot(main_rows, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in main_rows:
        x = _to_float(r["flops"]) / 1e9
        y1 = _to_float(r["accuracy"])
        y2 = _to_float(r["macro_f1"])
        color = "#4C72B0" if r["model"] == "eat_base" else "#55A868"
        label = f"{r['dataset']}-{r['model']}"
        axes[0].scatter(x, y1, color=color, s=50)
        axes[0].annotate(label, (x, y1), textcoords="offset points", xytext=(4, 4), fontsize=8)
        axes[1].scatter(x, y2, color=color, s=50)
        axes[1].annotate(label, (x, y2), textcoords="offset points", xytext=(4, 4), fontsize=8)

    axes[0].set_xlabel("FLOPs (G)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("FLOPs vs Accuracy")
    axes[1].set_xlabel("FLOPs (G)")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("FLOPs vs Macro-F1")
    fig.suptitle("Performance-Complexity Tradeoff")
    _save(fig, out_dir / "performance_complexity_tradeoff.png")


def _gain_plot(rel_rows, out_dir: Path):
    metrics = [
        ("params_reduction_pct", "Params↓%"),
        ("size_reduction_pct", "Size↓%"),
        ("flops_reduction_pct", "FLOPs↓%"),
        ("latency_reduction_pct", "Latency↓%"),
    ]
    ds_list = [r["dataset"] for r in rel_rows]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    vals0 = [_to_float(next(r for r in rel_rows if r["dataset"] == ds_list[0])[k]) for k, _ in metrics]
    vals1 = [_to_float(next(r for r in rel_rows if r["dataset"] == ds_list[1])[k]) for k, _ in metrics]
    ax.bar(x - w / 2, vals0, width=w, label=ds_list[0], color="#4C72B0")
    ax.bar(x + w / 2, vals1, width=w, label=ds_list[1], color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("Reduction (%)")
    ax.set_title("UT-EAT Reduction Summary vs EAT-base")
    ax.legend(fontsize=8)

    # throughput gain text on top-right
    gain_text = "\n".join(
        f"{r['dataset']}: throughput ×{float(r['throughput_gain_x']):.2f}" for r in rel_rows
    )
    ax.text(1.01, 0.98, gain_text, transform=ax.transAxes, va="top", fontsize=8)

    _save(fig, out_dir / "lightweighting_gain_summary.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/lightweighting_analysis"))
    p.add_argument("--figures-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/figures"))
    return p.parse_args()


def main():
    args = parse_args()
    main_rows = _read_csv(args.analysis_dir / "lightweighting_main_table.csv")
    rel_rows = _read_csv(args.analysis_dir / "lightweighting_relative_table.csv")

    _norm_plot(main_rows, args.figures_dir)
    _efficiency_plot(main_rows, args.figures_dir)
    _tradeoff_plot(main_rows, args.figures_dir)
    _gain_plot(rel_rows, args.figures_dir)
    print(f"Wrote figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
