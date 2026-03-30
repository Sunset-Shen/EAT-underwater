#!/usr/bin/env python3
"""Prepare ShipsEar manifests for EAT fine-tuning.

Output files (under output dir):
- train.tsv / valid.tsv / test.tsv
- train.lbl / valid.lbl / test.lbl
- label_descriptors.csv
- label_mapping.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class Sample:
    relpath: str
    frames: int
    label_name: str


def get_num_frames(path: Path) -> int:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            return int(wf.getnframes())

    try:
        import soundfile as sf

        info = sf.info(str(path))
        return int(info.frames)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read frame length for non-wav file: {path}. "
            "Install pysoundfile or keep files as wav."
        ) from exc


def discover_classes(data_root: Path, class_dirs: Sequence[str] | None) -> List[str]:
    if class_dirs:
        names = [d.strip() for d in class_dirs if d.strip()]
    else:
        names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if not names:
        raise ValueError(f"No class directories found under {data_root}")
    return names


def collect_samples(data_root: Path, classes: Sequence[str]) -> Dict[str, List[Sample]]:
    by_class: Dict[str, List[Sample]] = {}
    for cls in classes:
        cls_dir = data_root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")

        samples: List[Sample] = []
        for fp in sorted(cls_dir.rglob("*")):
            if not fp.is_file() or fp.suffix.lower() not in AUDIO_EXTS:
                continue
            frames = get_num_frames(fp)
            rel = fp.relative_to(data_root).as_posix()
            samples.append(Sample(relpath=rel, frames=frames, label_name=cls))

        if not samples:
            raise ValueError(f"No audio files found in class dir: {cls_dir}")
        by_class[cls] = samples
    return by_class


def split_class_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    items = list(samples)
    rng.shuffle(items)
    n = len(items)

    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid

    if n_train <= 0:
        n_train = 1
    if n_valid <= 0 and n >= 3:
        n_valid = 1
    if n_test <= 0 and n >= 2:
        n_test = 1

    while n_train + n_valid + n_test > n:
        if n_train >= n_valid and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_valid >= n_test and n_valid > 0:
            n_valid -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    train = items[:n_train]
    valid = items[n_train : n_train + n_valid]
    test = items[n_train + n_valid :]

    if not test and valid:
        test.append(valid.pop())
    elif not test and train:
        test.append(train.pop())

    if not valid and train:
        valid.append(train.pop())

    return train, valid, test


def write_tsv(path: Path, data_root: Path, samples: Sequence[Sample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{data_root.as_posix()}\n")
        for s in samples:
            f.write(f"{s.relpath}\t{s.frames}\n")


def write_lbl(path: Path, samples: Sequence[Sample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(samples):
            f.write(f"{idx}\t{s.label_name}\n")


def write_label_descriptors(path: Path, classes: Sequence[str]) -> Dict[str, int]:
    mapping = {cls: i for i, cls in enumerate(classes)}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for cls, idx in mapping.items():
            writer.writerow([idx, cls])
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShipsEar manifests for EAT")
    parser.add_argument("--data-root", type=Path, required=True, help="ShipsEar root dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output manifest dir")
    parser.add_argument(
        "--class-dirs",
        nargs="*",
        default=None,
        help="Optional explicit class directory names. If omitted, all subdirs under data root are used.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    classes = discover_classes(data_root, args.class_dirs)
    by_class = collect_samples(data_root, classes)

    rng = random.Random(args.seed)
    train_samples: List[Sample] = []
    valid_samples: List[Sample] = []
    test_samples: List[Sample] = []

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

    summary = {
        "classes": len(classes),
        "train": len(train_samples),
        "valid": len(valid_samples),
        "test": len(test_samples),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
