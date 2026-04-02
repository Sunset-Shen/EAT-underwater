#!/usr/bin/env python3
"""Helpers to adapt fairseq samples to model forward kwargs for compare_v1 scripts."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch


def to_device(sample: Any, device: torch.device):
    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, list):
        return [to_device(v, device) for v in sample]
    return sample


def _forward_param_names(model) -> list[str]:
    sig = inspect.signature(model.forward)
    names = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            names.append(p.name)
    return names


def extract_forward_kwargs(sample: Dict[str, Any], model) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "net_input" not in sample or not isinstance(sample["net_input"], dict):
        raise KeyError("sample['net_input'] missing or invalid")

    net_input = sample["net_input"]
    param_names = _forward_param_names(model)

    # do not feed supervision labels during inference/logits pass
    blocked = {"label", "target"}

    kwargs = {
        k: v
        for k, v in net_input.items()
        if k in param_names and k not in blocked and torch.is_tensor(v)
    }

    # fallback for unexpected naming (keep minimal assumptions)
    if not kwargs:
        for candidate in ["imgs", "source", "input", "x"]:
            v = net_input.get(candidate)
            if torch.is_tensor(v):
                kwargs = {candidate: v}
                break

    if not kwargs:
        for k, v in net_input.items():
            if torch.is_tensor(v) and k not in blocked:
                kwargs = {k: v}
                break

    if not kwargs:
        raise RuntimeError(f"Cannot derive model inputs from net_input keys={list(net_input.keys())}")

    debug = {
        "sample_keys": list(sample.keys()),
        "net_input_keys": list(net_input.keys()),
        "forward_param_names": param_names,
        "selected_input_keys": list(kwargs.keys()),
    }
    return kwargs, debug


def get_label_tensor(sample: Dict[str, Any]) -> torch.Tensor:
    if "label" in sample and torch.is_tensor(sample["label"]):
        return sample["label"]
    if "net_input" in sample and isinstance(sample["net_input"], dict):
        v = sample["net_input"].get("label")
        if torch.is_tensor(v):
            return v
    raise KeyError("label tensor not found in sample['label'] or sample['net_input']['label']")


def run_model_logits(model, sample: Dict[str, Any]):
    kwargs, debug = extract_forward_kwargs(sample, model)
    out = model(**kwargs)
    if isinstance(out, dict):
        raise RuntimeError(
            "Model returned dict instead of logits tensor. "
            "This usually means label/target was passed to forward. "
            f"selected_input_keys={debug.get('selected_input_keys')}"
        )
    return out, kwargs, debug
