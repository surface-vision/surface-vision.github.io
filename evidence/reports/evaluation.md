# Surface defect detector -- evaluation report

**Checkpoint** `yolov8n_neudet/best.pt` (`models/yolov8n_neudet/weights/best.pt`)
**Device** mps | **imgsz** 256 (served size; trained at 320) | **NMS IoU** 0.45 | **match IoU** 0.5
**Held-out split** `test` (180 images) | **tuning split** `val` (180 images)
**Generated** 2026-09-09T23:50:10+00:00

The threshold below was chosen on `val` and applied unchanged to
`test`. Nothing in this report is tuned on the held-out split.

> **Read this alongside its companion reports, and prefer them where they overlap.**
> The conf=0.05 used from section 2 onward is *this* module's
> recommendation, priced against spurious extra boxes on frames that already carried a
> defect. Every NEU-DET image does, so that is a nuisance-box rate, not a clean-steel
> rate. `reports/false_alarm.md` measures the clean-steel event and selects the
> **conf = 0.15** the console actually runs at; `reports/model_study.md` owns the input
> size and the model comparison, with bootstrap intervals this report does not compute;
> `reports/calibration.md` owns what the confidence number means.

## 1. Detection metrics -- `test` split

Produced by the ultralytics validator on its own protocol defaults
(conf 0.001, NMS IoU 0.7, max_det 300),
so these numbers are directly comparable with published NEU-DET results.

| metric | value |
|---|---|
| mAP50 | **0.7524** |
| mAP50-95 | **0.3967** |
| precision (mean over classes) | 0.6960 |
| recall (mean over classes) | 0.6867 |

| class | instances | AP50 | AP50-95 | precision | recall | base severity |
|---|---|---|---|---|---|---|
| crazing | 79 | 0.4443 | 0.1870 | 0.6881 | 0.2235 | high |
| inclusion | 89 | 0.8272 | 0.4517 | 0.7476 | 0.7865 | critical |
| patches | 99 | 0.9486 | 0.6223 | 0.8647 | 0.8889 | medium |
| pitted_surface | 46 | 0.7558 | 0.4273 | 0.6566 | 0.7174 | high |
| rolled-in_scale | 69 | 0.6295 | 0.2768 | 0.5665 | 0.5942 | high |
| scratches | 64 | 0.9093 | 0.4153 | 0.6525 | 0.9097 | medium |

## 2. Image-level classification -- `test` split @ conf=0.05

Each image is scored by its dominant (highest-confidence) detection, which is
exactly what `InferenceResult.dominant_class` hands the UI. Images where nothing
survives the threshold are counted as `(no detection)` rather than dropped.

- accuracy **98.9%** over 180 images
- macro-F1 **0.989** (weighted-F1 0.989)
- 0 images produced no detection at all
- 12 images carry more than one defect type; their image label is the
  majority class by instance count, ties broken by boxed area

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| crazing | 30 | 1.000 | 1.000 | 1.000 |
| inclusion | 30 | 1.000 | 1.000 | 1.000 |
| patches | 31 | 0.939 | 1.000 | 0.969 |
| pitted_surface | 29 | 1.000 | 0.931 | 0.964 |
| rolled-in_scale | 30 | 1.000 | 1.000 | 1.000 |
| scratches | 30 | 1.000 | 1.000 | 1.000 |

Matrix: `confusion_matrix.png`

## 3. False alarm analysis -- `val` split

Two levels of accounting. **Box level** (precision / recall / F1 / FP-per-image)
says whether the boxes on the HMI can be trusted. **Image level**
(detection rate / false alarm rate) prices the two events that actually cost
money: a coil that ships with an undetected defect, and a coil pulled for a
defect that is not there.

| conf | box recall | img detect rate | box precision | F1 | FP/img | false alarm rate | img class acc | cost/img |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.816 | 0.950 | 0.472 | 0.598 | 2.07 | 0.817 | 0.989 | 1.417 |
| 0.10 | 0.748 | 0.911 | 0.579 | 0.652 | 1.23 | 0.683 | 0.989 | 1.750 |
| 0.15 | 0.713 | 0.878 | 0.658 | 0.685 | 0.84 | 0.561 | 0.972 | 2.028 |
| 0.20 | 0.679 | 0.828 | 0.705 | 0.692 | 0.64 | 0.483 | 0.933 | 2.550 |
| 0.25 | 0.667 | 0.817 | 0.743 | 0.703 | 0.52 | 0.417 | 0.900 | 2.617 |
| 0.30 | 0.640 | 0.778 | 0.781 | 0.704 | 0.41 | 0.328 | 0.839 | 2.994 |
| 0.35 | 0.618 | 0.761 | 0.808 | 0.700 | 0.33 | 0.278 | 0.817 | 3.144 |
| 0.40 | 0.593 | 0.750 | 0.852 | 0.699 | 0.23 | 0.206 | 0.800 | 3.206 |
| 0.45 | 0.569 | 0.706 | 0.872 | 0.688 | 0.19 | 0.172 | 0.761 | 3.706 |
| 0.50 | 0.544 | 0.700 | 0.892 | 0.676 | 0.15 | 0.144 | 0.733 | 3.744 |
| 0.55 | 0.505 | 0.683 | 0.903 | 0.648 | 0.12 | 0.117 | 0.711 | 3.917 |
| 0.60 | 0.466 | 0.650 | 0.927 | 0.620 | 0.08 | 0.083 | 0.667 | 4.283 |
| 0.65 | 0.397 | 0.572 | 0.931 | 0.557 | 0.07 | 0.067 | 0.583 | 5.200 |
| 0.70 | 0.350 | 0.528 | 0.980 | 0.516 | 0.02 | 0.017 | 0.528 | 5.683 |
| 0.75 | 0.282 | 0.439 | 0.991 | 0.439 | 0.01 | 0.006 | 0.433 | 6.739 |
| 0.80 | 0.226 | 0.372 | 1.000 | 0.368 | 0.00 | 0.000 | 0.361 | 7.533 |
| 0.85 | 0.135 | 0.244 | 1.000 | 0.238 | 0.00 | 0.000 | 0.233 | 9.067 |
| 0.90 | 0.051 | 0.106 | 1.000 | 0.098 | 0.00 | 0.000 | 0.106 | 10.733 |
| 0.95 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | 12.000 |

### Recommended operating point: **conf = 0.05**

At conf=0.05 the detector finds the real defect on 95.0% of defective images while raising a spurious box on 81.7% of them (2.07 false boxes per image). Pricing one escaped defect at 12 unnecessary re-inspections, that is an expected 1.417 re-inspection-equivalents per inspected coil, the minimum over every threshold meeting the 90% detection floor the line will accept. The F1-optimal threshold 0.30 would lift box precision from 0.47 to 0.78 and cut false alarms from 81.7% to 32.8%, but its detection rate falls from 95.0% to 77.8% -- 4.4x as many coils shipping with an undetected defect. That is the trade F1 hides by weighting a miss and a false alarm equally. The optimum sits at the bottom edge of the swept 0.05-0.95 range, so the true minimum may lie outside it; read this as 'as far as the sweep allows', not as an interior optimum.

Cost model: one escaped defect = 12 unnecessary re-inspections.
An escaped defect downgrades a ~20 t coil from prime to secondary and can turn
into a customer claim; a false alarm costs the operator a manual look and a few
minutes of line availability. The recommendation is not fragile to that exact
number -- the same rule at other ratios gives:

| miss:false-alarm | threshold | img detect rate | false alarm rate | cost/img | clears floor |
|---|---|---|---|---|---|
| 3:1 | 0.15 | 0.878 | 0.561 | 0.928 | no |
| 6:1 | 0.05 | 0.950 | 0.817 | 1.117 | yes |
| 12:1 | 0.05 | 0.950 | 0.817 | 1.417 | yes |
| 25:1 | 0.05 | 0.950 | 0.817 | 2.067 | yes |
| 50:1 | 0.05 | 0.950 | 0.817 | 3.317 | yes |

Those rows minimise cost *without* the detection floor, to show where the
economics alone push the knob; the recommendation above additionally enforces the
floor.

For reference the F1-optimal threshold is 0.30; F1 weights a missed
defect and a false alarm equally, which is not how a mill is paid, so it is
reported but not followed.

Written to `operating_point.evaluate.json`.

**This is not the threshold the console runs at, and it should not be.** The false
alarm rate priced above is *a spurious extra box on a frame that already carried a
defect* -- every NEU-DET image does -- so it measures nuisance boxes on coils that
were going to be flagged anyway, not the event a mill buys on. `src/false_alarm.py`
measures the clean-steel rate on crops carrying no annotated defect and selects
**conf = 0.15**, which is what `reports/operating_point.json` holds and what the
console loads. This module writes its own file and will not touch that one without
`--update-operating-point --force-operating-point`.

## 4. Curves

- `pr_curves.png`
- `f1_vs_threshold.png`
- `false_alarms_vs_recall.png`

## 5. Per-class error gallery

Worst missed defects and most confident false positives per class, at the
recommended threshold, on the `test` split. Dotted green is ground
truth; the solid box is the failure.

- `0_crazing.png`
- `1_inclusion.png`
- `2_patches.png`
- `3_pitted_surface.png`
- `4_rolled-in_scale.png`
- `5_scratches.png`

Instance-level failure counts on `test` at conf=0.05 (IoU 0.5 match):

| class | base severity | instances | missed | miss rate | false positives | false positives landed on |
|---|---|---|---|---|---|---|
| crazing | high | 79 | 20 | 25.3% | 122 | crazing x122 |
| inclusion | critical | 89 | 13 | 14.6% | 60 | inclusion x57, patches x2, scratches x1 |
| patches | medium | 99 | 8 | 8.1% | 42 | patches x34, pitted_surface x8 |
| pitted_surface | high | 46 | 10 | 21.7% | 37 | pitted_surface x36, patches x1 |
| rolled-in_scale | high | 69 | 13 | 18.8% | 92 | rolled-in_scale x91, scratches x1 |
| scratches | medium | 64 | 6 | 9.4% | 61 | scratches x60, inclusion x1 |

The last column attributes each false positive to the true defect type of the
image it appeared on, which turns a raw false-positive count into a statement a
metallurgist can act on: a class whose false positives cluster on one other
class is a texture confusion to fix with data, while one spread evenly across
all six is an under-trained head.

## 6. Caveats

- **No clean strip in the data.** Every NEU-DET image contains a defect, so the
  false alarm rate reported here is "a spurious box on a coil that was already
  defective". The rate that matters on the line -- flagging a genuinely clean
  strip -- cannot be estimated from this dataset and must be measured on
  production frames before the threshold is committed.
- **Image-level labels are a simplification.** 12 of
  180 images carry more than one defect type; the confusion matrix
  collapses those to their majority class.
- **200x200 crops, not strip captures.** NEU-DET frames are single-defect crops.
  Full-width strip behaviour is exercised by `predict_tiled` in
  `src/inference.py` and is not measured here.
- **Cost ratio is a placeholder for a real one.** 12:1 is a
  defensible order of magnitude, not a Jindal Stainless number. The sensitivity
  table above is the honest answer until the quality department supplies the
  actual downgrade and re-inspection costs.

## 7. Latency

Validator-reported, per image at imgsz 256 on mps: preprocess 0.17 ms, inference 1.69 ms, postprocess 2.88 ms.
End-to-end latency as served by `src/inference.py` is measured separately by the
inference benchmark; these figures are the validator's own accounting.
