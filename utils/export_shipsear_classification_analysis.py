#!/usr/bin/env python3
"""Export ShipsEar confusion matrix, per-class metrics and prediction details for baseline/UT-EAT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from compare_v1_batch_adapter import get_label_tensor, run_model_logits, to_device


def _resolve_user_dir(saved_cfg, default_user_dir: Path) -> str:
    common_cfg = getattr(saved_cfg, "common", None)
    user_dir = getattr(common_cfg, "user_dir", None) if common_cfg is not None else None
    return user_dir or str(default_user_dir)


def _load_label_names(manifest_dir: Path) -> list[str]:
    labels = {}
    with (manifest_dir / "label_descriptors.csv").open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            labels[int(row[0])] = row[1]
    return [labels[i] for i in sorted(labels)]


def _evaluate_checkpoint(checkpoint: Path, manifest_dir: Path, subset: str, batch_size: int, device: str, user_dir: Path):
    from fairseq import checkpoint_utils, tasks, utils as fairseq_utils

    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint), {})
    saved_cfg = state.get("cfg", None)
    if saved_cfg is None:
        raise RuntimeError(f"Checkpoint missing cfg: {checkpoint}")

    fairseq_utils.import_user_module(SimpleNamespace(user_dir=_resolve_user_dir(saved_cfg, user_dir)))
    saved_cfg.task.data = str(manifest_dir)

    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(subset)
    dataset = task.dataset(subset)

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)], task=task)
    model = models[0].to(torch.device(device)).eval()

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_sentences=batch_size,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=42,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        disable_iterator_cache=True,
    ).next_epoch_itr(shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for sample in itr:
            if not sample:
                continue
            sample = to_device(sample, torch.device(device))
            logits, _, _ = run_model_logits(model, sample)
            pred = torch.argmax(logits, dim=-1)
            target = torch.argmax(get_label_tensor(sample), dim=-1)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(target.detach().cpu().tolist())

    return np.array(y_true), np.array(y_pred)


def _write_confusion_csv(path: Path, cm: np.ndarray, class_names: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *class_names])
        for i, name in enumerate(class_names):
            w.writerow([name, *cm[i].tolist()])


def _write_per_class_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for idx, name in enumerate(class_names):
        class_support = int(support[idx])
        class_acc = float(cm[idx, idx] / class_support) if class_support > 0 else 0.0
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "per_class_accuracy": class_acc,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": class_support,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "y_pred", "true_name", "pred_name", "correct"])
        for i, (t, p) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            w.writerow([i, t, p, class_names[t], class_names[p], int(t == p)])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, default=Path("/hy-tmp/exp/eat/manifests/shipsear"))
    p.add_argument("--baseline-checkpoint", type=Path, default=Path("/hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_best.pt"))
    p.add_argument("--uteat-checkpoint", type=Path, default=Path("/hy-tmp/exp/eat/runs/shipsear_uteat/checkpoint_best.pt"))
    p.add_argument("--subset", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--analysis-dir", type=Path, default=Path("/hy-tmp/exp/eat/results/compare_v1/shipsear/analysis"))
    p.add_argument("--user-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return p.parse_args()


def main():
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_label_names(args.manifest_dir)

    configs = [
        ("eat_base", args.baseline_checkpoint),
        ("ut_eat", args.uteat_checkpoint),
    ]

    for model_tag, ckpt in configs:
        y_true, y_pred = _evaluate_checkpoint(
            checkpoint=ckpt,
            manifest_dir=args.manifest_dir,
            subset=args.subset,
            batch_size=args.batch_size,
            device=args.device,
            user_dir=args.user_dir,
        )
        labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        _write_confusion_csv(args.analysis_dir / f"shipsear_{model_tag}_confusion_matrix.csv", cm, class_names)
        _write_per_class_csv(args.analysis_dir / f"shipsear_{model_tag}_per_class_metrics.csv", y_true, y_pred, class_names)
        _write_predictions_csv(args.analysis_dir / f"shipsear_{model_tag}_predictions.csv", y_true, y_pred, class_names)

    meta = {
        "manifest_dir": str(args.manifest_dir),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "uteat_checkpoint": str(args.uteat_checkpoint),
        "subset": args.subset,
    }
    (args.analysis_dir / "shipsear_analysis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
