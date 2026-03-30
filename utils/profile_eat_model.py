#!/usr/bin/env python3
"""Profile checkpoint size/params and simple latency for EAT finetuned checkpoints."""

import argparse
import time
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_latency(model, device, batch_size, target_length, num_warmup, num_iters):
    x = torch.randn(batch_size, 1, target_length, 128, device=device)
    net_input = {"source": x}

    model.eval()
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(**net_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_iters):
            _ = model(**net_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    return dt / num_iters


def maybe_flops(model, batch_size, target_length, device):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None

    x = torch.randn(batch_size, 1, target_length, 128, device=device)
    net_input = {"source": x}
    with torch.inference_mode():
        flops = FlopCountAnalysis(model, net_input).total()
    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("This script requires torch runtime environment.")

    try:
        from fairseq import checkpoint_utils
    except Exception as exc:
        raise RuntimeError(
            "This script requires fairseq runtime environment."
        ) from exc

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.checkpoint)],
    )
    model = models[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = count_trainable_params(model)
    ckpt_mb = args.checkpoint.stat().st_size / 1024 / 1024
    latency = run_latency(
        model,
        device=device,
        batch_size=args.batch_size,
        target_length=args.target_length,
        num_warmup=args.warmup,
        num_iters=args.iters,
    )
    flops = maybe_flops(model, args.batch_size, args.target_length, device)

    print(f"checkpoint: {args.checkpoint}")
    print(f"model_name: {saved_cfg.model._name}")
    print(f"trainable_params: {params}")
    print(f"checkpoint_size_mb: {ckpt_mb:.2f}")
    print(f"latency_sec_per_iter: {latency:.6f}")
    if flops is None:
        print("flops: N/A (install fvcore for FLOPs)")
    else:
        print(f"flops: {flops:.0f}")


if __name__ == "__main__":
    main()
