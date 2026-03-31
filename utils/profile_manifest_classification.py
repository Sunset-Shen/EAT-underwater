#!/usr/bin/env python3
"""Profile EAT checkpoints with a fixed and fair protocol for compare_v1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from compare_v1_batch_adapter import extract_forward_kwargs, to_device


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_timing(model, kwargs, device, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**kwargs)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(**kwargs)
        _sync(device)
    return (time.perf_counter() - t0) / max(iters, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--subset", default="test")
    p.add_argument("--throughput-batch-size", type=int, default=8)
    p.add_argument("--fallback-throughput-batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision-mode", default="fp32", choices=["fp32"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--user-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return p.parse_args()


def _resolve_user_dir(saved_cfg, default_user_dir: Path) -> str:
    common_cfg = getattr(saved_cfg, "common", None)
    user_dir = getattr(common_cfg, "user_dir", None) if common_cfg is not None else None
    return user_dir or str(default_user_dir)


def _pick_batch(task, dataset, bs, num_workers=0):
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=None,
        max_sentences=bs,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=42,
        num_shards=1,
        shard_id=0,
        num_workers=num_workers,
        epoch=1,
        disable_iterator_cache=True,
    ).next_epoch_itr(shuffle=False)
    for sample in itr:
        if sample:
            return sample
    raise RuntimeError(f"No sample batch found for batch_size={bs}")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from fairseq import checkpoint_utils, tasks, utils as fairseq_utils

    state = checkpoint_utils.load_checkpoint_to_cpu(str(args.checkpoint), {})
    saved_cfg = state.get("cfg", None)
    if saved_cfg is None:
        raise RuntimeError(f"Checkpoint missing cfg: {args.checkpoint}")

    user_dir = _resolve_user_dir(saved_cfg, args.user_dir)
    fairseq_utils.import_user_module(SimpleNamespace(user_dir=user_dir))

    saved_cfg.task.data = str(args.manifest_dir)
    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(args.subset)
    dataset = task.dataset(args.subset)

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)], task=task)
    model = models[0].to(device).eval()

    sample_bs1 = to_device(_pick_batch(task, dataset, 1), device)
    kwargs_bs1, sample_debug = extract_forward_kwargs(sample_bs1, model)
    lat_s_per_sample = _run_timing(model, kwargs_bs1, device, args.warmup, args.iters)

    throughput_bs = args.throughput_batch_size
    oom_fallback = False
    try:
        sample_bst = to_device(_pick_batch(task, dataset, throughput_bs), device)
        kwargs_bst, _ = extract_forward_kwargs(sample_bst, model)
        thr_s_per_iter = _run_timing(model, kwargs_bst, device, args.warmup, args.iters)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        oom_fallback = True
        throughput_bs = args.fallback_throughput_batch_size
        sample_bst = to_device(_pick_batch(task, dataset, throughput_bs), device)
        kwargs_bst, _ = extract_forward_kwargs(sample_bst, model)
        thr_s_per_iter = _run_timing(model, kwargs_bst, device, args.warmup, args.iters)

    flops = None
    flops_method = "fvcore.FlopCountAnalysis via wrapper forward(selected_model_input)"
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch.nn as nn

        class _KwargWrapper(nn.Module):
            def __init__(self, model_, key_):
                super().__init__()
                self.model = model_
                self.key = key_

            def forward(self, x):
                return self.model(**{self.key: x})

        if len(kwargs_bs1) != 1:
            raise RuntimeError("FLOPs currently requires a single tensor model input.")
        input_key, input_tensor = next(iter(kwargs_bs1.items()))

        with torch.no_grad():
            wrapped = _KwargWrapper(model, input_key)
            flops = float(FlopCountAnalysis(wrapped, (input_tensor,)).total())
    except Exception as exc:
        flops_method += f" (unavailable: {exc.__class__.__name__})"

    result = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "finetuned_checkpoint": str(args.checkpoint),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "checkpoint_size_mb": float(args.checkpoint.stat().st_size / 1024.0 / 1024.0),
        "flops": flops,
        "flops_method": flops_method,
        "latency_ms_per_sample": float(lat_s_per_sample * 1000.0),
        "throughput_samples_per_sec": float(throughput_bs / max(thr_s_per_iter, 1e-12)),
        "batch_size_for_throughput": int(throughput_bs),
        "throughput_oom_fallback": oom_fallback,
        "device": str(device),
        "precision_mode": args.precision_mode,
        "subset": args.subset,
        "protocol": {
            "latency_batch_size": 1,
            "preloaded_input": True,
            "include_dataloader_io": False,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "sample_structure_debug": sample_debug,
    }

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
