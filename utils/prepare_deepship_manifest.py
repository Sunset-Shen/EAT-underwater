#!/usr/bin/env python3
"""Prepare DeepShip manifests for EAT baseline.

Thin wrapper of `prepare_shipsear_manifest.py` to keep dataset-specific entrypoint.
"""

from pathlib import Path
import argparse

from prepare_shipsear_manifest import (
    discover_classes,
    collect_samples,
    split_class_samples,
    write_tsv,
    write_lbl,
    write_label_descriptors,
)
import random
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DeepShip manifests for EAT")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-dirs", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    classes = discover_classes(data_root, args.class_dirs)
    by_class = collect_samples(data_root, classes)

    rng = random.Random(args.seed)
    train_samples, valid_samples, test_samples = [], [], []

    for cls in classes:
        tr, va, te = split_class_samples(
            by_class[cls],
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            rng=rng,
        )
        train_samples.extend(tr)
        valid_samples.extend(va)
        test_samples.extend(te)

    rng.shuffle(train_samples)
    rng.shuffle(valid_samples)
    rng.shuffle(test_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "train.tsv", data_root, train_samples)
    write_tsv(output_dir / "valid.tsv", data_root, valid_samples)
    write_tsv(output_dir / "test.tsv", data_root, test_samples)

    write_lbl(output_dir / "train.lbl", train_samples)
    write_lbl(output_dir / "valid.lbl", valid_samples)
    write_lbl(output_dir / "test.lbl", test_samples)

    mapping = write_label_descriptors(output_dir / "label_descriptors.csv", classes)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "classes": len(classes),
                "train": len(train_samples),
                "valid": len(valid_samples),
                "test": len(test_samples),
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
