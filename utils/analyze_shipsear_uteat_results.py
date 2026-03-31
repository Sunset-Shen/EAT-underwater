#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PRETRAIN_COLUMNS = [
    "loss",
    "loss_cls",
    "loss_recon",
    "loss_physical",
    "loss_IMAGE_regression",
    "target_var",
    "pred_var",
    "masked_pct",
    "num_updates",
]


@dataclass
class RunPaths:
    baseline_finetune: Path
    uteat_smoke_pretrain: Path
    uteat_formal_pretrain: Path
    uteat_finetune: Path


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def _extract_train_log_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if "{" not in line or "}" not in line:
            continue
        payload = line[line.find("{") : line.rfind("}") + 1]
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        rows.append(obj)
    return rows


def _normalize_pretrain_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        norm = {}
        norm["num_updates"] = row.get("num_updates", row.get("train_num_updates"))
        norm["loss"] = row.get("loss", row.get("train_loss"))
        for col in PRETRAIN_COLUMNS:
            if col not in norm:
                norm[col] = row.get(col)
        out.append(norm)
    return out


def _read_curves(run_dir: Path) -> list[dict]:
    for reader in [lambda: _read_csv(run_dir / "curves.csv"), lambda: _read_jsonl(run_dir / "curves.jsonl"), lambda: _extract_train_log_rows(run_dir / "train.log")]:
        rows = reader()
        if rows:
            return _normalize_pretrain_rows(rows)
    return []


def _find_best_accuracy(log_path: Path):
    best = None
    final = None
    final_update = None
    for obj in _extract_train_log_rows(log_path):
        valid_acc = _to_float(obj.get("valid_accuracy"))
        if valid_acc is not None:
            final = valid_acc
            u = obj.get("num_updates")
            if u is not None:
                try:
                    final_update = int(u)
                except (TypeError, ValueError):
                    pass
            best = valid_acc if best is None else max(best, valid_acc)
        best_acc = _to_float(obj.get("valid_best_accuracy"))
        if best_acc is not None:
            best = best_acc if best is None else max(best, best_acc)
    return best, final, final_update


def _checkpoint_size_mb(ckpt: Path):
    return ckpt.stat().st_size / (1024 * 1024) if ckpt.exists() else None


def _parse_profile_metrics(run_dir: Path):
    metrics = {"Params": None, "LatencyMs": None, "FLOPs": None}
    for p in [run_dir / "profile.json", run_dir / "profile_metrics.json", run_dir / "run_summary.json"]:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            obj = {}
        metrics["Params"] = obj.get("trainable_params", obj.get("params", metrics["Params"]))
        metrics["LatencyMs"] = obj.get("latency_ms", obj.get("latency", metrics["LatencyMs"]))
        metrics["FLOPs"] = obj.get("flops", obj.get("FLOPs", metrics["FLOPs"]))
    log_path = run_dir / "profile.log"
    if log_path.exists():
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
        pats = {
            "Params": r"trainable[_ ]params\s*[:=]\s*([\d\.eE\+\-]+)",
            "LatencyMs": r"latency(?:_ms)?\s*[:=]\s*([\d\.eE\+\-]+)",
            "FLOPs": r"flops\s*[:=]\s*([\d\.eE\+\-]+)",
        }
        for k, pat in pats.items():
            if metrics[k] is None:
                m = re.search(pat, txt, flags=re.IGNORECASE)
                if m:
                    metrics[k] = _to_float(m.group(1))
    return metrics


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


# --- tiny png plotting ---
def new_image(w, h, color=(255, 255, 255)):
    return [[list(color) for _ in range(w)] for _ in range(h)]


def draw_pixel(img, x, y, color):
    h = len(img)
    w = len(img[0]) if h else 0
    if 0 <= x < w and 0 <= y < h:
        img[y][x] = list(color)


def draw_line(img, x0, y0, x1, y1, color, thickness=1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for tx in range(-thickness // 2, thickness // 2 + 1):
            for ty in range(-thickness // 2, thickness // 2 + 1):
                draw_pixel(img, x0 + tx, y0 + ty, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def fill_rect(img, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(len(img), y1)):
        for x in range(max(0, x0), min(len(img[0]), x1)):
            img[y][x] = list(color)


def save_png(path: Path, img):
    h = len(img)
    w = len(img[0]) if h else 0
    raw = bytearray()
    for row in img:
        raw.append(0)
        for px in row:
            raw.extend(bytes(px))

    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def make_placeholder_png(path: Path):
    img = new_image(960, 540, (250, 250, 250))
    fill_rect(img, 80, 80, 880, 460, (255, 255, 255))
    draw_line(img, 80, 80, 880, 80, (80, 80, 80), 2)
    draw_line(img, 80, 460, 880, 460, (80, 80, 80), 2)
    draw_line(img, 80, 80, 80, 460, (80, 80, 80), 2)
    draw_line(img, 880, 80, 880, 460, (80, 80, 80), 2)
    draw_line(img, 120, 420, 840, 120, (220, 120, 120), 3)
    save_png(path, img)


def make_line_png(path: Path, x, ys: list[tuple[list[float], tuple[int, int, int]]]):
    img = new_image(960, 540, (255, 255, 255))
    left, right, top, bottom = 90, 900, 50, 480
    draw_line(img, left, bottom, right, bottom, (0, 0, 0), 2)
    draw_line(img, left, top, left, bottom, (0, 0, 0), 2)
    xv = [v for v in x if v is not None]
    yv = [v for series, _ in ys for v in series if v is not None]
    if not xv or not yv:
        return make_placeholder_png(path)
    xmin, xmax = min(xv), max(xv)
    ymin, ymax = min(yv), max(yv)
    if xmax == xmin:
        xmax += 1.0
    if ymax == ymin:
        ymax += 1.0

    for series, color in ys:
        pts = []
        for xv1, yv1 in zip(x, series):
            if xv1 is None or yv1 is None:
                continue
            px = int(left + (xv1 - xmin) / (xmax - xmin) * (right - left))
            py = int(bottom - (yv1 - ymin) / (ymax - ymin) * (bottom - top))
            pts.append((px, py))
        for i in range(1, len(pts)):
            draw_line(img, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color, 2)
    save_png(path, img)


def make_bar_png(path: Path, values: list[float | None]):
    img = new_image(960, 540, (255, 255, 255))
    left, right, top, bottom = 90, 900, 50, 480
    draw_line(img, left, bottom, right, bottom, (0, 0, 0), 2)
    vals = [v for v in values if v is not None]
    if not vals:
        return make_placeholder_png(path)
    vmax = max(vals)
    if vmax == 0:
        vmax = 1.0
    bar_w = 180
    gap = 180
    x = 220
    colors = [(80, 130, 200), (90, 180, 120)]
    for i, v in enumerate(values[:2]):
        if v is None:
            x += bar_w + gap
            continue
        h = int((v / vmax) * (bottom - top - 10))
        fill_rect(img, x, bottom - h, x + bar_w, bottom, colors[i % len(colors)])
        x += bar_w + gap
    save_png(path, img)


def build_analysis(paths: RunPaths, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows = _read_curves(paths.uteat_smoke_pretrain)
    formal_rows = _read_curves(paths.uteat_formal_pretrain)

    merged = []
    for tag, rows in [("smoke", smoke_rows), ("formal", formal_rows)]:
        for r in rows:
            row = {"run": tag}
            for c in PRETRAIN_COLUMNS:
                row[c] = r.get(c, "")
            merged.append(row)
    write_csv(output_dir / "shipsear_uteat_pretrain_metrics.csv", merged, ["run", *PRETRAIN_COLUMNS])

    def extract_series(rows, col):
        out = []
        for r in rows:
            out.append(_to_float(r.get(col)))
        return out

    x_formal = extract_series(formal_rows, "num_updates")
    y_formal = extract_series(formal_rows, "loss")
    x_smoke = extract_series(smoke_rows, "num_updates")
    y_smoke = extract_series(smoke_rows, "loss")
    make_line_png(output_dir / "uteat_pretrain_total_loss_curve.png", x_formal if any(v is not None for v in x_formal) else x_smoke, [(y_formal, (31, 119, 180)), (y_smoke, (255, 127, 14))])

    comp_cols = ["loss_cls", "loss_recon", "loss_physical", "loss_IMAGE_regression"]
    ys = [(extract_series(formal_rows, c), color) for c, color in zip(comp_cols, [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189)])]
    make_line_png(output_dir / "uteat_pretrain_loss_components.png", x_formal, ys)

    finetune_rows = []
    for model_name, run_dir in [("EAT Baseline", paths.baseline_finetune), ("UT-EAT", paths.uteat_finetune)]:
        best, final, final_update = _find_best_accuracy(run_dir / "train.log")
        profile = _parse_profile_metrics(run_dir)
        best_ckpt = run_dir / "checkpoint_best.pt"
        last_ckpt = run_dir / "checkpoint_last.pt"
        finetune_rows.append(
            {
                "Model": model_name,
                "Best Valid Accuracy": best,
                "Final Valid Accuracy": final,
                "Final Update": final_update,
                "Best Checkpoint": str(best_ckpt) if best_ckpt.exists() else "",
                "Last Checkpoint": str(last_ckpt) if last_ckpt.exists() else "",
                "Checkpoint Size MB": _checkpoint_size_mb(last_ckpt),
                "Params": profile["Params"],
                "Latency (ms)": profile["LatencyMs"],
                "FLOPs": profile["FLOPs"],
                "Precision": "",
                "Recall": "",
                "F1": "",
                "Macro-F1": "",
            }
        )

    finetune_fields = list(finetune_rows[0].keys())
    write_csv(output_dir / "shipsear_baseline_vs_uteat_finetune.csv", finetune_rows, finetune_fields)
    eff_fields = ["Model", "Params", "Checkpoint Size MB", "Latency (ms)", "FLOPs", "Best Valid Accuracy"]
    write_csv(output_dir / "shipsear_model_efficiency_comparison.csv", finetune_rows, eff_fields)

    make_bar_png(output_dir / "shipsear_accuracy_comparison.png", [_to_float(finetune_rows[0]["Best Valid Accuracy"]), _to_float(finetune_rows[1]["Best Valid Accuracy"])])
    make_bar_png(output_dir / "shipsear_efficiency_comparison.png", [_to_float(finetune_rows[0]["Params"]), _to_float(finetune_rows[1]["Params"])])
    make_line_png(
        output_dir / "shipsear_accuracy_vs_latency.png",
        [_to_float(finetune_rows[0]["Latency (ms)"]), _to_float(finetune_rows[1]["Latency (ms)"])],
        [([_to_float(finetune_rows[0]["Best Valid Accuracy"]), _to_float(finetune_rows[1]["Best Valid Accuracy"])], (80, 130, 200))],
    )
    make_line_png(
        output_dir / "shipsear_accuracy_vs_params.png",
        [_to_float(finetune_rows[0]["Params"]), _to_float(finetune_rows[1]["Params"])],
        [([_to_float(finetune_rows[0]["Best Valid Accuracy"]), _to_float(finetune_rows[1]["Best Valid Accuracy"])], (80, 130, 200))],
    )

    notes = output_dir / "shipsear_uteat_analysis_notes.md"
    missing = []
    for label, p in [
        ("baseline finetune", paths.baseline_finetune),
        ("uteat smoke pretrain", paths.uteat_smoke_pretrain),
        ("uteat formal pretrain", paths.uteat_formal_pretrain),
        ("uteat finetune", paths.uteat_finetune),
    ]:
        if not p.exists():
            missing.append(f"- missing run directory: `{p}` ({label})")

    notes.write_text(
        "\n".join(
            [
                "# ShipsEar UT-EAT Analysis Notes",
                "",
                "## Data availability",
                *(missing if missing else ["- all configured run directories are present."]),
                "",
                "## Missing metrics handling",
                "- Precision/Recall/F1/Macro-F1 are kept empty when corresponding metrics are absent in logs.",
                "- Params/Latency/FLOPs are kept empty when no profile artifact is found in run directories.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "pretrain_rows": len(merged),
        "finetune_rows": len(finetune_rows),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-finetune", default="/hy-tmp/exp/eat/runs/shipsear_baseline")
    p.add_argument("--uteat-smoke-pretrain", default="/hy-tmp/exp/eat/runs/shipsear_uteat_smoke_pretrain")
    p.add_argument("--uteat-formal-pretrain", default="/hy-tmp/exp/eat/runs/shipsear_uteat_formal_pretrain")
    p.add_argument("--uteat-finetune", default="/hy-tmp/exp/eat/runs/shipsear_uteat")
    p.add_argument("--output-dir", default="/hy-tmp/exp/eat/results/shipsear_uteat_analysis")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    result = build_analysis(
        RunPaths(
            baseline_finetune=Path(a.baseline_finetune),
            uteat_smoke_pretrain=Path(a.uteat_smoke_pretrain),
            uteat_formal_pretrain=Path(a.uteat_formal_pretrain),
            uteat_finetune=Path(a.uteat_finetune),
        ),
        Path(a.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
