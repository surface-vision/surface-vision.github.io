#!/usr/bin/env python
"""Reference detections from the real Python model, for the browser parity check.

Runs the SHIPPED checkpoint through ultralytics at the shipped operating point
(conf 0.15, NMS IoU 0.45, imgsz 256) and dumps every detection as JSON. The browser
and Node paths in this directory must reproduce this file.

Also dumps the exact 256x256x3 uint8 tensor ultralytics built for each image, so a
disagreement can be attributed to preprocessing rather than guessed at.

Usage (from the project root):
    .venv/bin/python site/verify/reference_ultralytics.py \
        --out site/verify/parity_python.json \
        --dump-pixels /tmp/refpx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference import CLASS_NAMES, DEFAULT_IMGSZ  # noqa: E402

WEIGHTS = ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"
CONF = 0.15
IOU = 0.45

# 2 per class, first two of each class in the held-out test split, plus the strip
# frame the site ships as its aspect-ratio sample.
DEFAULT_IMAGES = [
    "crazing_271.jpg", "crazing_272.jpg",
    "inclusion_271.jpg", "inclusion_272.jpg",
    "patches_271.jpg", "patches_272.jpg",
    "pitted_surface_271.jpg", "pitted_surface_272.jpg",
    "rolled-in_scale_271.jpg", "rolled-in_scale_272.jpg",
    "scratches_271.jpg", "scratches_272.jpg",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site" / "verify" / "parity_python.json"))
    ap.add_argument("--dump-pixels", default=None,
                    help="directory to write the letterboxed uint8 tensors into")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    img_dir = ROOT / "data" / "neu-det" / "test" / "images"
    names = args.images or DEFAULT_IMAGES
    paths = [img_dir / n for n in names]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing image: {p}")

    model = YOLO(str(WEIGHTS))
    payload = {
        "weights": str(WEIGHTS),
        "conf": CONF,
        "iou": IOU,
        "imgsz": DEFAULT_IMGSZ,
        "class_names": CLASS_NAMES,
        "ultralytics": __import__("ultralytics").__version__,
        "images": [],
    }

    dump = Path(args.dump_pixels) if args.dump_pixels else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    for p in paths:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]

        res = model.predict(
            source=str(p), conf=CONF, iou=IOU, imgsz=DEFAULT_IMGSZ,
            device="cpu", verbose=False,
        )[0]

        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            dets.append({
                "class_id": int(box.cls.item()),
                "class_name": CLASS_NAMES[int(box.cls.item())],
                "raw": float(box.conf.item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        dets.sort(key=lambda d: -d["raw"])

        payload["images"].append({
            "name": p.name, "width": w, "height": h, "detections": dets,
        })

        if dump:
            # The tensor ultralytics itself built, recovered through the same
            # LetterBox the predictor uses, then converted BGR->RGB like preprocess.
            from ultralytics.data.augment import LetterBox
            lb = LetterBox((DEFAULT_IMGSZ, DEFAULT_IMGSZ), auto=False, stride=32)
            im = lb(image=bgr)[..., ::-1].copy()  # RGB uint8, HWC
            (dump / f"{p.stem}.rgb").write_bytes(im.tobytes())
            (dump / f"{p.stem}.meta.json").write_text(
                json.dumps({"shape": list(im.shape), "orig": [h, w]})
            )
            # And the raw original-resolution RGB, so Node can be fed identical
            # source pixels and the JPEG decoder ruled out as a variable.
            (dump / f"{p.stem}.orig.rgb").write_bytes(bgr[..., ::-1].copy().tobytes())

    Path(args.out).write_text(json.dumps(payload, indent=2))
    total = sum(len(i["detections"]) for i in payload["images"])
    print(f"wrote {args.out}: {len(payload['images'])} images, {total} detections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
