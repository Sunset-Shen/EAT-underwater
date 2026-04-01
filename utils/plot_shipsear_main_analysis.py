#!/usr/bin/env python3
"""Generate ShipsEar main-analysis figures for thesis section 4.3.3."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_compare_shipsear(compare_csv: Path):
    out = {}
    with compare_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "shipsear":
                continue
            out[row["model"]] = row
    return out


def _save(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_overall(compare_csv: Path, out_dir: Path):
    rows = _read_compare_shipsear(compare_csv)
    models = ["eat_base", "ut_eat"]
    labels = ["EAT-base", "UT-EAT"]
    acc = [float(rows[m]["accuracy"]) for m in models]
    f1 = [float(rows[m]["macro_f1"]) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, acc, color=["#4C72B0", "#55A868"])
    axes[0].set_title("ShipsEar Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, f1, color=["#4C72B0", "#55A868"])
    axes[1].set_title("ShipsEar Macro-F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Macro-F1")
    for i, v in enumerate(f1):
        axes[1].text(i, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir / "shipsear_overall_metrics.png")


def _extract_json_payload(line: str):
    m = re.search(r"\{.*\}", line)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_log_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        d = _extract_json_payload(line)
        if isinstance(d, dict):
            rows.append(d)
    return rows


def _pick_updates(row: dict):
    for k in ("num_updates", "train_num_updates", "valid_num_updates"):
        if k in row:
            try:
                return int(float(row[k]))
            except Exception:
                pass
    return None


def _pick(row: dict, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return None


def _smooth(y, w=5):
    if len(y) < w:
        return y
    arr = np.array(y, dtype=float)
    if np.any(np.isnan(arr)):
        if np.all(np.isnan(arr)):
            return y
        arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same").tolist()


def _plot_pretrain_loss(log_path: Path, out_dir: Path):
    rows = _parse_log_rows(log_path)
    series = {
        "train_loss": [],
        "train_loss_cls": [],
        "train_loss_recon": [],
        "train_loss_physical": [],
        "train_loss_IMAGE_regression": [],
    }
    x = []
    for r in rows:
        u = _pick_updates(r)
        if u is None:
            continue
        vals = {
            "train_loss": _pick(r, ["train_loss", "loss"]),
            "train_loss_cls": _pick(r, ["train_loss_cls", "loss_cls"]),
            "train_loss_recon": _pick(r, ["train_loss_recon", "loss_recon"]),
            "train_loss_physical": _pick(r, ["train_loss_physical", "loss_physical"]),
            "train_loss_IMAGE_regression": _pick(r, ["train_loss_IMAGE_regression", "loss_IMAGE_regression"]),
        }
        if all(v is None for v in vals.values()):
            continue
        x.append(u)
        for k, v in vals.items():
            series[k].append(np.nan if v is None else v)

    metrics = [
        ("train_loss", "#1f77b4"),
        ("train_loss_cls", "#ff7f0e"),
        ("train_loss_recon", "#2ca02c"),
        ("train_loss_physical", "#d62728"),
        ("train_loss_IMAGE_regression", "#9467bd"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for idx, (k, color) in enumerate(metrics):
        ax = axes[idx]
        y = np.array(series[k], dtype=float)
        if np.all(np.isnan(y)):
            ax.set_title(f"{k} (no data)")
            ax.grid(alpha=0.2)
            continue
        ym = np.where(np.isnan(y), np.nanmean(y), y)
        ax.plot(x, _smooth(ym.tolist(), w=7), linewidth=1.4, color=color)
        ax.set_title(k)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.2)
    axes[4].set_xlabel("num_updates")
    axes[5].axis("off")
    fig.suptitle("ShipsEar UT-EAT Formal Pretrain Loss Components", fontsize=12)
    _save(fig, out_dir / "shipsear_uteat_pretrain_loss_curve.png")


def _extract_valid_series(log_path: Path):
    rows = _parse_log_rows(log_path)
    triples = []
    for r in rows:
        u = _pick_updates(r)
        a = _pick(r, ["valid_accuracy", "accuracy"])
        l = _pick(r, ["valid_loss", "loss"])
        if u is None or (a is None and l is None):
            continue
        triples.append((u, np.nan if a is None else a, np.nan if l is None else l))
    triples.sort(key=lambda x: x[0])
    xs = [t[0] for t in triples]
    acc = [t[1] for t in triples]
    vloss = [t[2] for t in triples]
    return xs, acc, vloss


def _plot_finetune_curves(baseline_log: Path, uteat_log: Path, out_dir: Path):
    xb, ab, lb = _extract_valid_series(baseline_log)
    xu, au, lu = _extract_valid_series(uteat_log)
    lb_s = _smooth(lb, w=5)
    lu_s = _smooth(lu, w=5)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(xb, ab, label="EAT-base", color="#4C72B0", linewidth=1.6)
    axes[0].plot(xu, au, label="UT-EAT", color="#55A868", linewidth=1.6)
    if len(ab) > 0:
        ib = int(np.nanargmax(np.array(ab, dtype=float)))
        axes[0].plot(xb[ib], ab[ib], marker="*", markersize=12, color="#4C72B0")
        axes[0].annotate(f"best {ab[ib]:.4f}", (xb[ib], ab[ib]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    if len(au) > 0:
        iu = int(np.nanargmax(np.array(au, dtype=float)))
        axes[0].plot(xu[iu], au[iu], marker="*", markersize=12, color="#55A868")
        axes[0].annotate(f"best {au[iu]:.4f}", (xu[iu], au[iu]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    axes[0].set_title("Valid Accuracy")
    axes[0].set_xlabel("num_updates")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)

    axes[1].plot(xb, lb_s, label="EAT-base (MA5)", color="#4C72B0", linewidth=1.6)
    axes[1].plot(xu, lu_s, label="UT-EAT (MA5)", color="#55A868", linewidth=1.6)
    axes[1].set_title("Valid Loss (moving-average, window=5)")
    axes[1].set_xlabel("num_updates")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    _save(fig, out_dir / "shipsear_finetune_curves.png")


def _read_matrix(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    class_names = rows[0][1:]
    mat = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=float)
    return class_names, mat


def _plot_confusions(cm_base_csv: Path, cm_ut_csv: Path, out_dir: Path):
    names1, m1 = _read_matrix(cm_base_csv)
    names2, m2 = _read_matrix(cm_ut_csv)
    names = names1 if len(names1) <= len(names2) else names2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [
        (axes[0], m1, "EAT-base Confusion Matrix"),
        (axes[1], m2, "UT-EAT Confusion Matrix"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        vmax = np.max(mat) if mat.size > 0 else 0
        thresh = vmax * 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(round(mat[i, j]))
                txt_color = "white" if mat[i, j] > thresh else "black"
                ax.text(j, i, f"{val:d}", ha="center", va="center", color=txt_color, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir / "shipsear_confusion_matrices.png")


def _read_per_class(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["class_name"] for r in rows]
    acc = [float(r["per_class_accuracy"]) for r in rows]
    return names, acc


def _plot_per_class_accuracy(base_csv: Path, ut_csv: Path, out_dir: Path):
    names, a1 = _read_per_class(base_csv)
    names2, a2 = _read_per_class(ut_csv)
    if names != names2:
        raise RuntimeError("Class order mismatch between baseline and UT-EAT per-class metrics")

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w / 2, a1, width=w, label="EAT-base", color="#4C72B0")
    ax.bar(x + w / 2, a2, width=w, label="UT-EAT", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class Accuracy")
    ax.set_title("ShipsEar Per-class Accuracy Comparison")
    ax.legend(fontsize=9)
    _save(fig, out_dir / "shipsear_per_class_accuracy.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compare-main-csv", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv"))
    p.add_argument("--pretrain-log", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/shipsear_formal_pretrain_2026-03-31_164952.log"))
    p.add_argument("--baseline-finetune-log", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/train_shipsear_eatbase_2026-03-31_115626.log"))
    p.add_argument("--uteat-finetune-log", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/shipsear_uteat_finetune_2026-04-01_005407.log"))
    p.add_argument("--analysis-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/shipsear/analysis"))
    p.add_argument("--figures-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/figures"))
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_overall(args.compare_main_csv, args.figures_dir)
    _plot_pretrain_loss(args.pretrain_log, args.figures_dir)
    _plot_finetune_curves(args.baseline_finetune_log, args.uteat_finetune_log, args.figures_dir)

    cm_base = args.analysis_dir / "shipsear_eat_base_confusion_matrix.csv"
    cm_ut = args.analysis_dir / "shipsear_ut_eat_confusion_matrix.csv"
    per_base = args.analysis_dir / "shipsear_eat_base_per_class_metrics.csv"
    per_ut = args.analysis_dir / "shipsear_ut_eat_per_class_metrics.csv"

    _plot_confusions(cm_base, cm_ut, args.figures_dir)
    _plot_per_class_accuracy(per_base, per_ut, args.figures_dir)

    print(f"Figures written to: {args.figures_dir}")


if __name__ == "__main__":
    main()
