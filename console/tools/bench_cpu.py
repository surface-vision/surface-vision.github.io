#!/usr/bin/env python3
"""Measure what this host actually does, through the vendored code, on CPU.

Produced every latency figure in README.md. Run it on the Space's own hardware to
replace them with that host's numbers -- the point of the script is that no figure
on the Space page has to be inherited from the machine the project was built on.

    python tools/bench_cpu.py [--threads N] [--repeats N]

`--threads` defaults to the CPUs this container is actually allowed to use, which
is what app.py does; pass 2 on a bigger machine to approximate the free tier.
Everything it touches is inside the Space: the vendored modules, the two committed
checkpoints, the 18 sample frames and the one strip capture.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

SPACE_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=0, help="0 = as app.py decides")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--weights", default="models/yolov8n_neudet/weights/best.pt")
    parser.add_argument("--cam", action="store_true", help="also time one EigenCAM")
    args = parser.parse_args(argv)

    try:
        allowed = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        source = "sched_getaffinity"
    except AttributeError:
        allowed = os.cpu_count() or 1
        source = "os.cpu_count"
    threads = args.threads or max(1, min(allowed, 8))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ.setdefault("YOLO_OFFLINE", "1")

    sys.path.insert(0, str(SPACE_ROOT / "vendor"))
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(threads)
    import inference
    import ood_guard

    print(f"host        : {allowed} CPU(s) via {source}, torch pinned to {threads} thread(s)")
    print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()} "
          f"mps={torch.backends.mps.is_available()}")

    detector = inference.load_detector(
        weights=str(SPACE_ROOT / args.weights), device="cpu",
        conf=0.15, iou=0.45, imgsz=inference.DEFAULT_IMGSZ,
    )
    print(f"detector    : {detector}")
    print(f"classes     : {len(detector.class_names)} -> {detector.class_names}")
    detector.warmup(3)

    def stats(label: str, times: list[float], extra: str = "") -> None:
        print(f"  {label:44s} mean {statistics.mean(times):7.1f} ms   "
              f"p50 {statistics.median(times):7.1f}   "
              f"p95 {np.percentile(times, 95):7.1f}   "
              f"{1000 / statistics.mean(times):6.1f} fps  {extra}")

    frames = sorted((SPACE_ROOT / "data/neu-det/test/images").glob("*.jpg"))
    print(f"\n--- {len(frames)} shipped NEU-DET frames, 200x200, one at a time ---")
    arrays = [np.asarray(Image.open(f).convert("RGB")) for f in frames]

    gate = []
    for arr in arrays:
        t0 = time.perf_counter(); ood_guard.inspect(arr); gate.append((time.perf_counter() - t0) * 1000)
    stats("ood_guard.inspect (numpy + OpenCV)", gate)

    single, detections = [], 0
    for _ in range(args.repeats):
        for arr in arrays:
            t0 = time.perf_counter(); r = detector.predict(arr); single.append((time.perf_counter() - t0) * 1000)
            detections += r.defect_count
    stats("DefectDetector.predict", single,
          f"{detections // args.repeats} boxes over {len(frames)} frames")

    batched = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        results = detector.predict_batch(arrays, batch_size=8)
        batched.append((time.perf_counter() - t0) * 1000 / len(arrays))
    stats("DefectDetector.predict_batch (per frame)", batched,
          f"{sum(r.defect_count for r in results)} boxes")

    strip_path = next(iter(sorted((SPACE_ROOT / "assets").glob("*.jpg"))), None)
    if strip_path is not None:
        strip = np.asarray(Image.open(strip_path).convert("RGB"))
        h, w = strip.shape[:2]
        print(f"\n--- {strip_path.name}, {w}x{h} real line-scan capture ---")
        whole, whole_n = [], 0
        for _ in range(max(3, args.repeats // 3)):
            t0 = time.perf_counter(); r = detector.predict(strip); whole.append((time.perf_counter() - t0) * 1000)
            whole_n = r.defect_count
        stats("predict, whole frame squashed to 256 px", whole, f"{whole_n} detections")
        tiled, tiled_n = [], 0
        for _ in range(max(3, args.repeats // 3)):
            t0 = time.perf_counter()
            r = detector.predict_tiled(strip, tile=256, overlap=0.2)
            tiled.append((time.perf_counter() - t0) * 1000)
            tiled_n = r.defect_count
        n_tiles = (len(inference._tile_origins(w, 256, int(256 * 0.8)))
                   * len(inference._tile_origins(h, 256, int(256 * 0.8))))
        stats("predict_tiled, 256 px tiles / 0.2 overlap", tiled,
              f"{tiled_n} detections over {n_tiles} tiles "
              f"({statistics.mean(tiled) / n_tiles:.1f} ms/tile)")

    if args.cam:
        from explain import explain as explain_frame
        print("\n--- EigenCAM (vendor/explain.py), first call then steady state ---")
        for i in range(3):
            t0 = time.perf_counter()
            exp = explain_frame(detector, arrays[0], method="auto")
            dt = (time.perf_counter() - t0) * 1000
            print(f"  call {i + 1}: {dt:8.0f} ms   method {exp.method}   layer {exp.layer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
