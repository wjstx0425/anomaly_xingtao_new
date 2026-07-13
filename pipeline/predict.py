from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from train_template_matcher import load_gray, similarity


def main():
    p = argparse.ArgumentParser(description="Predict one C789 crop with a trained template model.")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--view", choices=["left_top", "left_bottom"], required=True)
    p.add_argument("--slot", choices=[f"slot{i:02d}" for i in range(1, 7)], required=True)
    a = p.parse_args()
    model = json.loads((a.model_dir / "model.json").read_text(encoding="utf-8"))
    group = model["groups"][f"{a.view}/{a.slot}"]
    image = load_gray(a.image, model["width"])
    templates = [cv2.imread(str(a.model_dir / p), cv2.IMREAD_GRAYSCALE) for p in group["templates"]]
    score = max(similarity(image, t, model["shift"]) for t in templates)
    result = {
        "image": a.image,
        "view": a.view,
        "slot": a.slot,
        "similarity": score,
        "threshold": group["threshold"],
        "prediction": "defect" if score < group["threshold"] else "normal",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
