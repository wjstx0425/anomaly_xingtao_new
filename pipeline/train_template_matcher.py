from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate a multi-template matcher for C789.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--templates-per-slot", type=int, default=5)
    p.add_argument("--max-train-per-slot", type=int, default=120)
    p.add_argument("--shift", type=int, default=12)
    p.add_argument("--normal-quantile", type=float, default=0.995)
    return p.parse_args()


def discover(root: Path):
    rows = []
    pattern = re.compile(r"[\\/](top|bottom)[\\/](normal_test|normal|defect)[\\/]", re.I)
    slot_re = re.compile(r"slot(\d+)", re.I)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = pattern.search(str(p))
        sm = slot_re.search(str(p))
        if not m or not sm:
            continue
        position, label = m.group(1).lower(), m.group(2).lower()
        rows.append({
            "view": f"left_{position}",
            "slot": f"slot{int(sm.group(1)):02d}",
            "label": label,
            "split": "train" if label == "normal" else "test",
            "path": str(p.resolve()),
        })
    return sorted(rows, key=lambda x: (x["view"], x["slot"], x["label"], x["path"]))


def load_gray(path: str, width: int):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(image, (3, 3), 0)


def similarity(image, template, shift):
    if image.shape != template.shape:
        image = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(image, shift, shift, shift, shift, cv2.BORDER_REFLECT_101)
    response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    return float(response.max())


def choose_templates(paths, width, count, max_samples):
    if len(paths) > max_samples:
        idx = np.linspace(0, len(paths) - 1, max_samples, dtype=int)
        paths = [paths[i] for i in idx]
    images = [load_gray(p, width) for p in paths]
    thumb = np.stack([cv2.resize(im, (64, 32), interpolation=cv2.INTER_AREA).reshape(-1) for im in images]).astype(
        np.float32
    )
    thumb -= thumb.mean(axis=1, keepdims=True)
    thumb /= thumb.std(axis=1, keepdims=True) + 1e-6
    center = np.median(thumb, axis=0)
    first = int(np.argmin(np.mean((thumb - center) ** 2, axis=1)))
    selected = [first]
    min_dist = np.mean((thumb - thumb[first]) ** 2, axis=1)
    while len(selected) < min(count, len(images)):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.mean((thumb - thumb[nxt]) ** 2, axis=1))
    return [(paths[i], images[i]) for i in selected]


def best_score(path, templates, width, shift):
    image = load_gray(path, width)
    return max(similarity(image, t, shift) for t in templates)


def metrics(records):
    tp = sum(r["prediction"] == "defect" and r["label"] == "defect" for r in records)
    tn = sum(r["prediction"] == "normal" and r["label"] == "normal_test" for r in records)
    fp = sum(r["prediction"] == "defect" and r["label"] == "normal_test" for r in records)
    fn = sum(r["prediction"] == "normal" and r["label"] == "defect" for r in records)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
    }


def main():
    args = parse_args()
    rows = discover(args.data)
    if not rows:
        raise SystemExit("No C789 images discovered")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dataset_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    model = {
        "method": "cv2.TM_CCOEFF_NORMED",
        "width": args.width,
        "shift": args.shift,
        "normal_quantile": args.normal_quantile,
        "groups": {},
    }
    predictions = []
    groups = sorted({(r["view"], r["slot"]) for r in rows})
    for view, slot in groups:
        group = [r for r in rows if r["view"] == view and r["slot"] == slot]
        train = [r["path"] for r in group if r["label"] == "normal"]
        validation = [r for r in group if r["label"] == "normal_test"]
        defects = [r for r in group if r["label"] == "defect"]
        selected = choose_templates(train, args.width, args.templates_per_slot, args.max_train_per_slot)
        template_dir = args.output / "templates" / view / slot
        template_dir.mkdir(parents=True, exist_ok=True)
        templates, template_files = [], []
        for i, (source, image) in enumerate(selected, 1):
            out = template_dir / f"template_{i:02d}.png"
            cv2.imwrite(str(out), image)
            templates.append(image)
            template_files.append(str(out.relative_to(args.output)))
        normal_scores = [best_score(r["path"], templates, args.width, args.shift) for r in validation]
        threshold = float(np.quantile(normal_scores, 1.0 - args.normal_quantile)) if normal_scores else 0.8
        key = f"{view}/{slot}"
        model["groups"][key] = {
            "threshold": threshold,
            "templates": template_files,
            "template_sources": [x[0] for x in selected],
            "train_count": len(train),
            "normal_test_count": len(validation),
            "defect_count": len(defects),
        }
        for r in validation + defects:
            score = best_score(r["path"], templates, args.width, args.shift)
            predictions.append({
                **r,
                "similarity": score,
                "threshold": threshold,
                "prediction": "defect" if score < threshold else "normal",
            })
        print(f"{key}: threshold={threshold:.5f}, train={len(train)}, test={len(validation)}, defect={len(defects)}")

    with (args.output / "model.json").open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=predictions[0].keys())
        w.writeheader()
        w.writerows(predictions)
    report = {"overall": metrics(predictions)}
    for view in sorted({r["view"] for r in predictions}):
        report[view] = metrics([r for r in predictions if r["view"] == view])
    with (args.output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
