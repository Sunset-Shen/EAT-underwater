#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def _plot_overall(main_rows, fig_dir: Path):
    variants = [r["variant"] for r in main_rows]
    acc = [float(r["accuracy"]) for r in main_rows]
    f1 = [float(r["macro_f1"]) for r in main_rows]
    x = np.arange(len(variants))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, acc, color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=20)
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, f1, color="#55A868")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants, rotation=20)
    axes[1].set_title("Macro-F1")
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("ShipsEar Ablation Overall Metrics")
    _save(fig, fig_dir / "ablation_overall_metrics.png")


def _plot_module_gain(delta_rows, fig_dir: Path):
    keep = [r for r in delta_rows if r["variant"] in {"ut_mask", "ut_loss", "ut_full"}]
    x = np.arange(len(keep))
    w = 0.36
    dacc = [float(r["accuracy_delta_vs_ut_backbone"]) for r in keep]
    df1 = [float(r["macro_f1_delta_vs_ut_backbone"]) for r in keep]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, dacc, width=w, label="ΔAccuracy", color="#4C72B0")
    ax.bar(x + w / 2, df1, width=w, label="ΔMacro-F1", color="#55A868")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["variant"] for r in keep])
    ax.set_ylabel("Delta vs UT-backbone")
    ax.set_title("Module Gain Comparison")
    ax.legend(fontsize=8)
    _save(fig, fig_dir / "ablation_module_gain.png")


def _parse_log(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "{" not in line:
            continue
        try:
            d = json.loads(line[line.find("{"):line.rfind("}")+1])
        except Exception:
            continue
        if "valid_accuracy" in d:
            u = d.get("valid_num_updates", d.get("num_updates", d.get("train_num_updates")))
            if u is not None:
                rows.append((int(float(u)), float(d["valid_accuracy"])))
    rows.sort(key=lambda x: x[0])
    return rows


def _plot_finetune_curves(fig_dir: Path):
    logs = {
        "ut_backbone": Path("/hy-tmp/exp/eat/runs/ablation_shipsear_ut_backbone_finetune/train.log"),
        "ut_mask": Path("/hy-tmp/exp/eat/runs/ablation_shipsear_ut_mask_finetune/train.log"),
        "ut_loss": Path("/hy-tmp/exp/eat/runs/ablation_shipsear_ut_loss_finetune/train.log"),
        "ut_full": Path("/hy-tmp/exp/eat/runs/ablation_shipsear_ut_full_finetune/train.log"),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, path in logs.items():
        if not path.exists():
            continue
        rows = _parse_log(path)
        if not rows:
            continue
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        ax.plot(x, y, label=name)
    ax.set_title("Ablation Finetune Valid Accuracy Curves")
    ax.set_xlabel("num_updates")
    ax.set_ylabel("valid_accuracy")
    ax.legend(fontsize=8)
    _save(fig, fig_dir / "ablation_finetune_curves.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/ablation_shipsear"))
    p.add_argument("--figures-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/figures"))
    return p.parse_args()


def main():
    args = parse_args()
    main_rows = _read_csv(args.ablation_dir / "ablation_main_table.csv")
    delta_rows = _read_csv(args.ablation_dir / "ablation_delta_table.csv")

    _plot_overall(main_rows, args.figures_dir)
    _plot_module_gain(delta_rows, args.figures_dir)
    _plot_finetune_curves(args.figures_dir)
    print(f"Wrote ablation figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
