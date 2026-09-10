# surface-vision.github.io

The live demo, the evidence behind it, and the Python console source, for an AI
surface defect detection system for stainless steel strip.

**Open it: <https://surface-vision.github.io>** — no install, no sign-in, nothing
uploaded. The page *is* the detector: the shipped 6.22 MB YOLOv8n checkpoint,
exported to a 12.1 MB ONNX graph, running on the WASM backend of ONNX Runtime
inside your own browser.

## What is in here

| | |
|---|---|
| `index.html`, `css/`, `js/` | The demo page. `js/detect.js` is the pre/post-processing, `js/tiling.js` the wide-strip path, `js/calibration.js` the isotonic calibrator's knots. |
| `model/` | The ONNX export the page runs. 12,128,540 bytes. |
| `samples/` | Twelve held-out NEU-DET test frames plus two 2048x1000 GC10 line-scan strip chips. |
| `verify/` | The parity harness and its output. |
| `evidence/` | 14 report narratives, 28 JSON artefacts and the deck — the files the experiments wrote, not summaries. Index at [`evidence/`](https://surface-vision.github.io/evidence/). |
| `console/` | The Streamlit operator console in full, with both checkpoints, so running it locally is one command. Index at [`console/`](https://surface-vision.github.io/console/). |

## The numbers the page states

Every figure below is traceable to a file in `evidence/reports/`.

| | measured | source |
|---|---|---|
| mAP50, held-out test split (180 frames, never trained or tuned on) | **0.7524** | `model_study.json` |
| mAP50-95 / precision / recall | 0.3967 / 0.696 / 0.687 | `model_study.json` |
| Image-level classification accuracy | 98.9% (178 of 180) | `evaluation.json` |
| Joint checkpoint, same split | mAP50 **0.7642**, clean-frame false alarms 93.7% → 32.5%, cross-domain ROC AUC 0.608 → 0.958 | `gap1_cross_domain.json`, `gap1_detection_metrics.json` |
| Calibration error (ECE), before → after isotonic | 0.1418 → 0.0461, mAP unchanged | `calibration.json` |
| Browser vs the PyTorch reference | **40 of 40** detections matched, max box delta 0.14 px, max confidence delta 0.0016 | `verify/parity_browser.json` |
| Browser inference | 30–50 ms per frame, single-threaded WASM (medians of 31.7 ms and 49.1 ms in two recorded runs) | `verify/parity_browser.json`, `verify/live_demo_check.json` |

## What this page deliberately does not do

It runs the six-class NEU-DET head, not the better ten-class joint checkpoint, so
the cross-domain false alarms are live here. It has no out-of-scope gate — that is
`console/vendor/ood_guard.py`, and it was not ported, so this page will score a
photograph of anything. It has no coil verdict, no PLC and no MES. The limitations
panel on the page says all of this, at length.

The browser build demonstrates that the artefact is small and portable enough to
run where the data is. It is not a proposal to inspect coil in a browser tab.

## Reproducing the parity check

```
cd verify && npm install
npm run reference          # dumps the Python ultralytics reference
npm run parity && npm run compare
node browser_parity.mjs --url https://surface-vision.github.io/
```

A capability demonstration on public research datasets (NEU-DET, GC10-DET,
Severstal), not a Jindal Stainless production system. No line data was used.
