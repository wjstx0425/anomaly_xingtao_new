"""Subtract a measured fixed reflection template, retaining invalid-region evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def subtract_template(image: np.ndarray, template: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped BGR difference and signed residual in stored pixel units."""
    if image.shape != template.shape or image.dtype != np.uint8 or template.dtype != np.uint8:
        raise ValueError("image and template must share shape and uint8 dtype")
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("image and template must be nonempty BGR images")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    signed = image.astype(np.float32) - scale * template.astype(np.float32)
    return np.rint(np.clip(signed, 0, 255)).astype(np.uint8), signed


def read_image(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image, hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    image, image_hash = read_image(args.image)
    template, template_hash = read_image(args.template)
    corrected, signed = subtract_template(image, template, args.scale)
    conservative, _ = subtract_template(image, template, .8 * args.scale)
    source_clip = np.all(image == 255, axis=2)
    source_nearclip = np.any(image >= 250, axis=2)
    template_nearclip = np.any(template >= 250, axis=2)
    below_zero = np.any(signed < 0, axis=2)
    uncertain = source_nearclip | template_nearclip
    warning = corrected.copy()
    warning[uncertain] = (0, 0, 255)
    variants = {"original":image, "reflection_template":template, "corrected":corrected,
                "subtract_80pct":conservative, "corrected_saturation_overlay":warning}
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {}

    def save(name: str, value: np.ndarray) -> None:
        ok, data = cv2.imencode(".png", value)
        if not ok:
            raise OSError(f"cannot encode {name}")
        raw = data.tobytes()
        with (args.output / name).open("xb") as stream:
            stream.write(raw)
        outputs[name] = hashlib.sha256(raw).hexdigest()

    save("source_all255_mask.png", source_clip.astype(np.uint8)*255)
    save("saturation_uncertainty_mask.png", uncertain.astype(np.uint8)*255)
    save("negative_residual_mask.png", below_zero.astype(np.uint8)*255)
    # Persist signed differences: PNG clipping must not hide over-subtraction evidence.
    np.savez_compressed(args.output / "signed_residual.npz", residual=signed)
    panels, cards = [], []
    for name, value in variants.items():
        save(name + ".png", value)
        thumb = cv2.resize(value, (600, round(value.shape[0]*600/value.shape[1])), interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb,28,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240))
        cv2.putText(thumb,name,(8,20),0,.5,(0,0,0),1)
        save("preview_" + name + ".png",thumb)
        panels.append(thumb)
        cards.append(f'<figure><a href="{name}.png"><img src="preview_{name}.png"></a><figcaption>{name}</figcaption></figure>')
    cv2.imwrite(str(args.output / "before_after.jpg"),np.hstack([panels[0],panels[2]]))
    # Same unmodified coordinates for an enlarged stamp comparison.
    if image.shape[:2] == (3036,4024):
        crops=[]
        for name in ("original","corrected","corrected_saturation_overlay"):
            crop=variants[name][1300:2030,1590:2090]
            crop=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240))
            cv2.putText(crop,name,(8,20),0,.45,(0,0,0),1)
            crops.append(crop)
        save("stamp_comparison.png",np.hstack(crops))
    metadata = {
        "diagnostic_only": True, "image": str(args.image.resolve()), "template":str(args.template.resolve()),
        "image_sha256":image_hash,"template_sha256":template_hash,"shape_hwc":list(image.shape),
        "scale":args.scale,"registration":"none; same sensor coordinates, no image warping",
        "formula":"round(clip(image - scale*template,0,255)) in saved pixel values",
        "source_all255_pct":float(source_clip.mean()*100),
        "source_any_ge250_pct":float(source_nearclip.mean()*100),
        "template_any_ge250_pct":float(template_nearclip.mean()*100),
        "negative_any_channel_pct":float(below_zero.mean()*100),
        "limitations":"Unknown exposure/gain/gamma; template assumed stable. Saturated source areas can become inverted template ghosts after subtraction. Background texture also subtracted. No recovered-surface or inspection-accuracy claim.",
        "files_sha256":outputs,
    }
    for key,path in (("image",args.image),("template",args.template)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[key+"_sha256"]:
            raise RuntimeError(f"{key} changed during processing")
    (args.output/"report.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output/"index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>实测反光模板扣除</title><style>body{font-family:sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(2,1fr)}img{width:100%}figure{margin:8px}</style>'
        '<h1>实测反光模板扣除</h1><p>corrected：1×指定比例扣除；subtract_80pct：0.8×指定比例扣除。无配准、锐化、去噪或修补。点击查看全分辨率。</p>'
        '<p>红色为原图或模板任一通道≥250的区域，不能因扣除后不白就认为细节恢复。饱和区扣除模板可能产生反向亮环。此结果仅作诊断，不是检测合格图。</p>'
        '<p><a href="report.json">参数与统计</a> | <a href="saturation_uncertainty_mask.png">饱和不确定区</a> | <a href="negative_residual_mask.png">负残差区域</a></p>'
        '<div class="grid">'+''.join(cards)+'</div>',encoding="utf-8")
    print(args.output/"index.html")


if __name__ == "__main__":
    main()
