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


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    f = _to_float(v)
    return int(f) if f is not None else None


@dataclass
class RunPaths:
    baseline_finetune: Path
    uteat_smoke_pretrain: Path
    uteat_formal_pretrain: Path
    uteat_finetune: Path


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
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


def _kv_extract(line: str, key: str):
    # supports: valid_accuracy=98.0 / valid_accuracy: 98.0 / valid_accuracy 98.0
    m = re.search(rf"{re.escape(key)}\s*(?:=|:)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
    return _to_float(m.group(1)) if m else None


def _parse_log_payload(line: str):
    if "{" in line and "}" in line:
        payload = line[line.find("{") : line.rfind("}") + 1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass

    keys = [
        "valid_accuracy",
        "valid_best_accuracy",
        "valid_num_updates",
        "num_updates",
        "train_num_updates",
        "loss",
        "train_loss",
        "loss_cls",
        "loss_recon",
        "loss_physical",
        "loss_IMAGE_regression",
        "target_var",
        "pred_var",
        "masked_pct",
    ]
    out = {}
    for k in keys:
        v = _kv_extract(line, k)
        if v is not None:
            out[k] = v
    return out if out else None


def _extract_train_log_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = _parse_log_payload(line)
        if payload:
            rows.append(payload)
    return rows


def _normalize_pretrain_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        norm = {c: row.get(c) for c in PRETRAIN_COLUMNS}
        norm["num_updates"] = row.get("num_updates", row.get("train_num_updates", row.get("valid_num_updates")))
        norm["loss"] = row.get("loss", row.get("train_loss", row.get("valid_loss")))
        out.append(norm)
    return out


def _select_structured_rows(run_dir: Path, source_trace: list[str]) -> tuple[list[dict], str]:
    candidates = [
        (run_dir / "curves.csv", _read_csv),
        (run_dir / "curves.jsonl", _read_jsonl),
        (run_dir / "train.log", _extract_train_log_rows),
    ]
    for path, reader in candidates:
        rows = reader(path)
        if rows:
            source_trace.append(f"pretrain rows from {path}")
            return _normalize_pretrain_rows(rows), str(path)
    source_trace.append(f"pretrain rows missing for {run_dir}")
    return [], ""


def _read_pretrain_rows(run_name: str, run_dir: Path, source_trace: list[str]) -> list[dict]:
    rows, _ = _select_structured_rows(run_dir, source_trace)
    cleaned = []
    excluded = 0
    for r in rows:
        u = _to_int(r.get("num_updates"))
        if run_name == "smoke" and u is not None and u > 100:
            excluded += 1
            continue
        rr = {"run": run_name}
        for c in PRETRAIN_COLUMNS:
            rr[c] = r.get(c, "")
        cleaned.append(rr)
    if excluded:
        source_trace.append(f"excluded {excluded} smoke rows with num_updates>100")
    return cleaned


def _accuracy_rows_from_structured(run_dir: Path, source_trace: list[str]) -> list[dict]:
    for path, reader in [
        (run_dir / "curves.csv", _read_csv),
        (run_dir / "curves.jsonl", _read_jsonl),
        (run_dir / "train.log", _extract_train_log_rows),
    ]:
        rows = reader(path)
        valid = []
        for r in rows:
            acc = _to_float(r.get("valid_accuracy"))
            if acc is None:
                continue
            update = (
                _to_int(r.get("valid_num_updates"))
                or _to_int(r.get("num_updates"))
                or _to_int(r.get("train_num_updates"))
            )
            valid.append({"valid_accuracy": acc, "update": update, "valid_best_accuracy": _to_float(r.get("valid_best_accuracy"))})
        if valid:
            source_trace.append(f"finetune accuracy rows from {path}")
            return valid
    source_trace.append(f"finetune accuracy rows missing for {run_dir}")
    return []


def _find_finetune_summary(run_dir: Path, source_trace: list[str]):
    run_summary = _read_json(run_dir / "run_summary.json") or {}
    best_from_summary = _to_float(run_summary.get("best_valid_accuracy"))
    best_update_from_summary = _to_int(run_summary.get("best_valid_update"))
    if run_summary:
        source_trace.append(f"used run_summary: {run_dir / 'run_summary.json'}")

    acc_rows = _accuracy_rows_from_structured(run_dir, source_trace)
    best = best_from_summary
    final = None
    final_update = None
    best_update = best_update_from_summary

    if acc_rows:
        final_row = max(acc_rows, key=lambda x: (-1 if x["update"] is None else x["update"]))
        final = final_row["valid_accuracy"]
        final_update = final_row["update"]

        for r in acc_rows:
            cand = r["valid_best_accuracy"] if r["valid_best_accuracy"] is not None else r["valid_accuracy"]
            if cand is None:
                continue
            if best is None or cand > best:
                best = cand
                best_update = r["update"]

    best_ckpt = run_summary.get("best_checkpoint") or str(run_dir / "checkpoint_best.pt")
    last_ckpt = run_summary.get("last_checkpoint") or str(run_dir / "checkpoint_last.pt")

    if not Path(best_ckpt).exists():
        best_ckpt = ""
    if not Path(last_ckpt).exists():
        last_ckpt = ""

    return {
        "best": best,
        "final": final,
        "final_update": final_update,
        "best_update": best_update,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
    }


def _parse_profile_metrics(run_dir: Path, source_trace: list[str]):
    metrics = {"Params": None, "LatencyMs": None, "FLOPs": None, "Checkpoint Size MB": None}

    json_candidates = [run_dir / "profile.json", run_dir / "profile_metrics.json", run_dir / "run_summary.json"]
    text_candidates = [run_dir / "profile.txt", run_dir / "profile.log"]

    for p in json_candidates:
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue
        source_trace.append(f"profile json from {p}")
        metrics["Params"] = _to_float(obj.get("trainable_params") or obj.get("params") or metrics["Params"])
        lat = _to_float(obj.get("latency_ms") or obj.get("latency") or obj.get("latency_sec_per_iter"))
        if lat is not None:
            if obj.get("latency_sec_per_iter") is not None:
                lat *= 1000.0
            metrics["LatencyMs"] = lat
        metrics["FLOPs"] = _to_float(obj.get("flops") or obj.get("FLOPs") or metrics["FLOPs"])
        metrics["Checkpoint Size MB"] = _to_float(obj.get("checkpoint_size_mb") or metrics["Checkpoint Size MB"])

    patterns = {
        "Params": r"trainable[_ ]params\s*[:=]\s*([\d\.eE\+\-]+)",
        "LatencyMs": r"latency(?:_ms)?\s*[:=]\s*([\d\.eE\+\-]+)",
        "LatencySec": r"latency_sec_per_iter\s*[:=]\s*([\d\.eE\+\-]+)",
        "FLOPs": r"flops\s*[:=]\s*([\d\.eE\+\-]+)",
        "CheckpointMB": r"checkpoint_size_mb\s*[:=]\s*([\d\.eE\+\-]+)",
    }

    for p in text_candidates:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        source_trace.append(f"profile text from {p}")
        if metrics["Params"] is None:
            m = re.search(patterns["Params"], txt, flags=re.IGNORECASE)
            if m:
                metrics["Params"] = _to_float(m.group(1))
        if metrics["LatencyMs"] is None:
            m = re.search(patterns["LatencyMs"], txt, flags=re.IGNORECASE)
            if m:
                metrics["LatencyMs"] = _to_float(m.group(1))
            else:
                m = re.search(patterns["LatencySec"], txt, flags=re.IGNORECASE)
                if m:
                    sec = _to_float(m.group(1))
                    metrics["LatencyMs"] = sec * 1000.0 if sec is not None else None
        if metrics["FLOPs"] is None:
            m = re.search(patterns["FLOPs"], txt, flags=re.IGNORECASE)
            if m:
                metrics["FLOPs"] = _to_float(m.group(1))
        if metrics["Checkpoint Size MB"] is None:
            m = re.search(patterns["CheckpointMB"], txt, flags=re.IGNORECASE)
            if m:
                metrics["Checkpoint Size MB"] = _to_float(m.group(1))

    return metrics


def _checkpoint_size_mb(ckpt_path: str):
    if not ckpt_path:
        return None
    p = Path(ckpt_path)
    return p.stat().st_size / (1024 * 1024) if p.exists() else None


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


def make_line_png(path: Path, x, ys):
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


def make_bar_png(path: Path, values):
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
    trace = []

    trace.append(f"using smoke run dir: {paths.uteat_smoke_pretrain}")
    trace.append(f"using formal run dir: {paths.uteat_formal_pretrain}")
    trace.append("excluded legacy run dir by default: /hy-tmp/exp/eat/runs/shipsear_uteat_pretrain")

    smoke_rows = _read_pretrain_rows("smoke", paths.uteat_smoke_pretrain, trace)
    formal_rows = _read_pretrain_rows("formal", paths.uteat_formal_pretrain, trace)
    merged = smoke_rows + formal_rows
    write_csv(output_dir / "shipsear_uteat_pretrain_metrics.csv", merged, ["run", *PRETRAIN_COLUMNS])

    def extract_series(rows, col):
        return [_to_float(r.get(col)) for r in rows]

    x_formal = extract_series(formal_rows, "num_updates")
    y_formal = extract_series(formal_rows, "loss")
    x_smoke = extract_series(smoke_rows, "num_updates")
    y_smoke = extract_series(smoke_rows, "loss")
    make_line_png(
        output_dir / "uteat_pretrain_total_loss_curve.png",
        x_formal if any(v is not None for v in x_formal) else x_smoke,
        [(y_formal, (31, 119, 180)), (y_smoke, (255, 127, 14))],
    )

    comp_cols = ["loss_cls", "loss_recon", "loss_physical", "loss_IMAGE_regression"]
    ys = [
        (extract_series(formal_rows, c), color)
        for c, color in zip(comp_cols, [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189)])
    ]
    make_line_png(output_dir / "uteat_pretrain_loss_components.png", x_formal, ys)

    finetune_rows = []
    for model_name, run_dir in [
        ("EAT Baseline", paths.baseline_finetune),
        ("UT-EAT", paths.uteat_finetune),
    ]:
        trace.append(f"using finetune run dir ({model_name}): {run_dir}")
        fin = _find_finetune_summary(run_dir, trace)
        profile = _parse_profile_metrics(run_dir, trace)
        ckpt_mb = profile["Checkpoint Size MB"]
        if ckpt_mb is None:
            ckpt_mb = _checkpoint_size_mb(fin["last_ckpt"])

        finetune_rows.append(
            {
                "Model": model_name,
                "Best Valid Accuracy": fin["best"],
                "Final Valid Accuracy": fin["final"],
                "Final Update": fin["final_update"],
                "Best Checkpoint": fin["best_ckpt"],
                "Last Checkpoint": fin["last_ckpt"],
                "Checkpoint Size MB": ckpt_mb,
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

    make_bar_png(
        output_dir / "shipsear_accuracy_comparison.png",
        [_to_float(finetune_rows[0]["Best Valid Accuracy"]), _to_float(finetune_rows[1]["Best Valid Accuracy"])],
    )
    make_bar_png(
        output_dir / "shipsear_efficiency_comparison.png",
        [_to_float(finetune_rows[0]["Params"]), _to_float(finetune_rows[1]["Params"])],
    )
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

    missing = []
    for label, p in [
        ("baseline finetune", paths.baseline_finetune),
        ("uteat smoke pretrain", paths.uteat_smoke_pretrain),
        ("uteat formal pretrain", paths.uteat_formal_pretrain),
        ("uteat finetune", paths.uteat_finetune),
    ]:
        if not p.exists():
            missing.append(f"- missing run directory: `{p}` ({label})")

    notes = output_dir / "shipsear_uteat_analysis_notes.md"
    note_lines = [
        "# ShipsEar UT-EAT Analysis Notes",
        "",
        "## Run directories used",
        f"- baseline finetune: `{paths.baseline_finetune}`",
        f"- uteat smoke pretrain: `{paths.uteat_smoke_pretrain}`",
        f"- uteat formal pretrain: `{paths.uteat_formal_pretrain}`",
        f"- uteat finetune: `{paths.uteat_finetune}`",
        "- legacy directory excluded by default: `/hy-tmp/exp/eat/runs/shipsear_uteat_pretrain`",
        "",
        "## Source trace",
        *[f"- {t}" for t in trace],
        "",
        "## Missing directories",
        *(missing if missing else ["- none"]),
        "",
        "## Missing metrics handling",
        "- Precision/Recall/F1/Macro-F1 remain empty when not found in artifacts.",
        "- Params/Latency/FLOPs remain empty only when profile artifacts are unavailable or unparsable.",
    ]
    notes.write_text("\n".join(note_lines), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "pretrain_rows": len(merged),
        "smoke_rows": len(smoke_rows),
        "formal_rows": len(formal_rows),
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
