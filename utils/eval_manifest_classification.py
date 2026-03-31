#!/usr/bin/env python3
"""Evaluate single-label classification checkpoints on EAT manifest splits.

Outputs a machine-readable JSON with accuracy/macro-f1 and traceable checkpoint fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _to_device(sample, device: torch.device):
    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: _to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_to_device(v, device) for v in sample]
    return sample


def _resolve_user_dir(saved_cfg, default_user_dir: Path) -> str:
    common_cfg = getattr(saved_cfg, "common", None)
    user_dir = getattr(common_cfg, "user_dir", None) if common_cfg is not None else None
    return user_dir or str(default_user_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--subset", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--user-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from fairseq import checkpoint_utils, tasks, utils as fairseq_utils

    state = checkpoint_utils.load_checkpoint_to_cpu(str(args.checkpoint), {})
    saved_cfg = state.get("cfg", None)
    if saved_cfg is None:
        raise RuntimeError(f"Checkpoint missing cfg: {args.checkpoint}")

    user_dir = _resolve_user_dir(saved_cfg, args.user_dir)
    fairseq_utils.import_user_module(SimpleNamespace(user_dir=user_dir))

    # ensure evaluated manifest is explicit and traceable
    saved_cfg.task.data = str(args.manifest_dir)

    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(args.subset)
    dataset = task.dataset(args.subset)

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)], task=task)
    model = models[0].to(torch.device(args.device)).eval()

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_sentences=args.batch_size,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=42,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
        epoch=1,
        disable_iterator_cache=True,
    ).next_epoch_itr(shuffle=False)

    y_true, y_pred = [], []
    device = torch.device(args.device)

    with torch.no_grad():
        for sample in itr:
            if not sample:
                continue
            sample = _to_device(sample, device)
            logits = model(sample["net_input"]["source"])
            pred = torch.argmax(logits, dim=-1)
            target = torch.argmax(sample["label"], dim=-1)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(target.detach().cpu().tolist())

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    result = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "subset": args.subset,
        "manifest_dir": str(args.manifest_dir),
        "init_checkpoint": args.init_checkpoint,
        "finetuned_checkpoint": str(args.checkpoint),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "num_samples": len(y_true),
        "batch_size": args.batch_size,
        "device": args.device,
    }

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
