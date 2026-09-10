# Cross-domain generalisation: does this detector work on other steel?

Generated 2026-09-09T16:12:22+05:30 by `src/cross_domain.py`.

`src/false_alarm.py` measures false alarms on defect-free *crops of NEU-DET*
images and says plainly that it is an upper bound from a dataset containing no
defect-free frame. This report replaces that proxy with real, verified
defect-free steel from a different line, and measures the clean/defective
separation on that same line so the two are comparable.

**Units.** A *crop* is one 200x200 tile: one inference, one operator alarm.
A *frame* is one 1600x256 strip frame, flagged if any of its tiles alarms.
Coil dispositions are built from frames, so the frame number is the one that
decides whether prime steel ships.

**Intervals.** Tiles are clustered inside frames, so every crop-level interval
is an image-clustered bootstrap (`false_alarm.cluster_bootstrap_interval`),
and the design effect is the factor by which a naive binomial interval would
have overstated precision. AUC intervals resample whole frames on both arms.

## Run `baseline`

- checkpoint: `models/yolov8n_neudet/weights/best.pt`
- sha256: `6661c7a09037137a...`
- imgsz 256, iou 0.45, device mps, crop 200 px (1.28x magnification)
- data: `<scratch>/severstal`
- scored 505 held-out defect-free frames (4040 tiles) and 191 defective frames (546 labelled defective tiles)
- withheld from scoring: 495 clean, 209 defective (hash split, frac 0.5, seed `jsw-ps1-cross-domain-holdout-v1`)
- clean manifest sha256 `1688fe2263c1df0c...`
- note: Severstal steel defect dataset (Voxel51/severstal_steel_defects), 1600x256 line-scan strip. The clean arm is the subset the dataset itself verifies as defect-free; the defective arm carries pixel masks.
- 982 tiles came from defective frames but carry no labelled pixels; they are excluded from both arms (see note below).

**Bottom line at the shipped operating point (conf 0.15, imgsz 256).** On steel with no defect in it, 93.7% of strip frames raise an alarm and the detector puts 2.02 boxes on the average tile. Defect recall on the same line is 72.0%, so the alarm carries AUC 0.608 [0.576, 0.640] of information against 0.968 [0.948, 0.984] in domain. The shipped disposition chain returns HOLD on a clean coil.

### False alarms on genuinely defect-free steel, and recall on the same domain

| conf | clean-crop FA [95% CI] | clean-FRAME FA [95% CI] | boxes/crop | defect crop recall [95% CI] | defect frame recall |
|---|---|---|---|---|---|
| 0.05 | 68.2% [65.5, 70.8] | 98.0% [96.6, 99.2] | 4.35 | 87.7% [84.4, 90.8] | 98.4% |
| 0.10 | 60.4% [57.6, 63.2] | 96.0% [94.3, 97.6] | 2.78 | 80.6% [76.6, 84.3] | 93.7% |
| 0.15 **<- shipped** | 55.1% [52.2, 58.0] | 93.7% [91.5, 95.6] | 2.02 | 72.0% [67.3, 76.4] | 89.0% |
| 0.20 | 50.1% [47.3, 53.1] | 91.1% [88.5, 93.5] | 1.57 | 63.9% [59.0, 69.0] | 81.2% |
| 0.25 | 44.4% [41.5, 47.3] | 86.7% [83.6, 89.5] | 1.24 | 57.0% [51.8, 62.5] | 74.9% |
| 0.30 | 38.2% [35.5, 40.9] | 85.1% [81.8, 88.1] | 0.98 | 51.8% [46.5, 57.2] | 70.2% |
| 0.40 | 29.4% [26.9, 32.0] | 77.4% [73.9, 81.0] | 0.61 | 41.8% [36.8, 46.7] | 63.4% |
| 0.50 | 23.5% [21.3, 25.8] | 71.7% [67.7, 75.6] | 0.40 | 30.2% [25.7, 34.8] | 50.8% |
| 0.60 | 16.3% [14.4, 18.2] | 57.8% [53.5, 62.2] | 0.23 | 18.3% [14.7, 22.1] | 36.6% |
| 0.70 | 8.9% [7.6, 10.2] | 39.8% [35.2, 44.4] | 0.11 | 8.2% [5.9, 10.8] | 18.3% |
| 0.80 | 3.0% [2.4, 3.6] | 17.4% [14.1, 20.6] | 0.03 | 2.7% [1.4, 4.2] | 7.3% |

Design effect on the crop false alarm rate ranges 1.48-3.51: treating 4040 tiles as independent draws would have quoted an interval up to 1.87x too narrow.

### What the false positives are called, at the shipped conf 0.15

| class | false boxes | share |
|---|---|---|
| inclusion | 3556 | 43.5% |
| patches | 2807 | 34.3% |
| scratches | 849 | 10.4% |
| pitted_surface | 524 | 6.4% |
| rolled-in_scale | 414 | 5.1% |
| crazing | 26 | 0.3% |

Total 8176 boxes on 4040 tiles of steel that carries no defect at all.

For scale: the 982 tiles cut from defective frames but holding no labelled pixels alarm at 48.0%. They are excluded from both arms, but the similarity of that rate to the verified-clean rate is itself evidence that the detector is not responding to the defects.

`inclusion` is the class `report.DispositionRules.hold_classes` treats as
zero-tolerance: a single frame carrying it holds the coil unconditionally.

### Separation: is the confidence score measuring defects or measuring domain?

| population | AUC (max box confidence) | 95% CI | positives | negatives |
|---|---|---|---|---|
| cross-domain (Severstal, defective vs verified clean) | 0.608 | [0.576, 0.640] | 546 | 4040 |
| in-domain (NEU-DET test, defect vs mined clean) | 0.968 | [0.948, 0.984] | 197 | 771 |

The two mined arms are not balanced across crop sizes -- containment needs
room, so large crops are over-represented among the positives and small
crops among the negatives. Stratifying by crop size and recombining with
Mann-Whitney weights removes that composition effect:

| crop px | positives | negatives | AUC |
|---|---|---|---|
| 60 | 40 | 448 | 0.954 |
| 80 | 52 | 212 | 0.968 |
| 100 | 63 | 77 | 0.962 |
| 120 | 42 | 34 | 0.954 |

Size-stratified in-domain AUC: **0.960** (pooled 0.968). Either way the in-domain figure sits far above the cross-domain one, and the gap is not a crop-size artefact.

The in-domain negatives are mined crops that clear every labelled box by 6 px at sizes [60, 80, 100, 120] px -- NEU-DET has no defect-free image, so this proxy is the best available and it
flatters the model. Magnification is held equal across the two rows (1.28x); field of view is not, and cannot be, because a 200x200 NEU-DET frame has no room for a 200 px defect-free window.

### Shipped disposition chain on a clean coil -- 200x200 tiles (deployment geometry)

```
coil SEVERSTAL-CLEAN-HOLDOUT-BASELINE: 480 frames, defect rate 55.2%, 960 detections, p95 severity 89.0 -> HOLD
  - Defect rate 55.2% exceeds the hold limit of 25.0%.
  - 95th-percentile severity 89.0 reaches the hold limit of 70.0.
  - 62 frame(s) in the critical severity band (limit 0).
  - 101 frame(s) contain 'inclusion', a zero-tolerance defect (limit 0) - it cannot be removed downstream.
```

### Shipped disposition chain on a clean coil -- whole 1600x256 strip frames

```
coil SEVERSTAL-CLEAN-HOLDOUT-BASELINE-WHOLEFRAME: 60 frames, defect rate 50.0%, 53 detections, p95 severity 72.0 -> HOLD
  - Defect rate 50.0% exceeds the hold limit of 25.0%.
  - 95th-percentile severity 72.0 reaches the hold limit of 70.0.
  - 3 frame(s) in the critical severity band (limit 0).
  - 7 frame(s) contain 'inclusion', a zero-tolerance defect (limit 0) - it cannot be removed downstream.
```

Every frame on that coil is verified defect-free steel, so every detection
is a false alarm and every trigger listed above fired on nothing.

---

## Run `audit_negatives_only`

- checkpoint: `<scratch>/audit_runs/v8n_neg/weights/best.pt`
- sha256: `f1836bcc0f36ede6...`
- imgsz 256, iou 0.45, device mps, crop 200 px (1.28x magnification)
- data: `<scratch>/severstal`
- scored 417 held-out defect-free frames (3336 tiles) and 191 defective frames (546 labelled defective tiles)
- withheld from scoring: 583 clean, 209 defective (hash split, frac 0.5, seed `jsw-ps1-cross-domain-holdout-v1`)
- clean manifest sha256 `3311712120c6b19e...`, 180 clean frame(s) removed by an explicit training manifest
- note: Held-out arm of the same Severstal pull, with the 180 source frames the audit's negatives-only warm-start actually trained on removed by explicit manifest (--exclude-list). This checkpoint lives in the audit scratchpad, not in models/; it is scored here to show the harness discriminates between checkpoints and to record the negatives-only trap.
- 982 tiles came from defective frames but carry no labelled pixels; they are excluded from both arms (see note below).

**Bottom line at the shipped operating point (conf 0.15, imgsz 256).** On steel with no defect in it, 6.2% of strip frames raise an alarm and the detector puts 0.01 boxes on the average tile. Defect recall on the same line is 3.7%, so the alarm carries AUC 0.567 [0.545, 0.593] of information against 0.964 [0.941, 0.983] in domain. The shipped disposition chain returns ACCEPT on a clean coil.

### False alarms on genuinely defect-free steel, and recall on the same domain

| conf | clean-crop FA [95% CI] | clean-FRAME FA [95% CI] | boxes/crop | defect crop recall [95% CI] | defect frame recall |
|---|---|---|---|---|---|
| 0.05 | 1.9% [1.3, 2.5] | 10.8% [7.9, 13.9] | 0.03 | 8.4% [5.2, 12.3] | 16.8% |
| 0.10 | 1.4% [0.9, 1.9] | 8.6% [6.0, 11.3] | 0.02 | 5.3% [2.8, 8.3] | 10.5% |
| 0.15 **<- shipped** | 1.0% [0.6, 1.3] | 6.2% [4.1, 8.6] | 0.01 | 3.7% [1.5, 6.5] | 6.8% |
| 0.20 | 0.8% [0.4, 1.2] | 5.0% [3.1, 7.2] | 0.01 | 3.7% [1.5, 6.5] | 6.8% |
| 0.25 | 0.6% [0.3, 0.9] | 3.8% [2.2, 5.8] | 0.01 | 3.1% [1.2, 5.5] | 6.3% |
| 0.30 | 0.4% [0.2, 0.7] | 3.1% [1.4, 5.0] | 0.00 | 1.6% [0.6, 2.9] | 4.2% |
| 0.40 | 0.2% [0.1, 0.4] | 1.9% [0.7, 3.4] | 0.00 | 0.9% [0.2, 1.9] | 2.1% |
| 0.50 | 0.1% [0.0, 0.2] | 0.7% [0.0, 1.7] | 0.00 | 0.2% [0.0, 0.6] | 0.5% |
| 0.60 | 0.1% [0.0, 0.1] | 0.5% [0.0, 1.2] | 0.00 | 0.0% [0.0, 0.0] | 0.0% |
| 0.70 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.00 | 0.0% [0.0, 0.0] | 0.0% |
| 0.80 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.00 | 0.0% [0.0, 0.0] | 0.0% |

Design effect on the crop false alarm rate ranges 0.99-1.65: treating 3336 tiles as independent draws would have quoted an interval up to 1.28x too narrow.

### What the false positives are called, at the shipped conf 0.15

| class | false boxes | share |
|---|---|---|
| patches | 19 | 55.9% |
| scratches | 6 | 17.6% |
| rolled-in_scale | 4 | 11.8% |
| pitted_surface | 4 | 11.8% |
| inclusion | 1 | 2.9% |

Total 34 boxes on 3336 tiles of steel that carries no defect at all.

For scale: the 982 tiles cut from defective frames but holding no labelled pixels alarm at 0.9%. They are excluded from both arms, but the similarity of that rate to the verified-clean rate is itself evidence that the detector is not responding to the defects.

`inclusion` is the class `report.DispositionRules.hold_classes` treats as
zero-tolerance: a single frame carrying it holds the coil unconditionally.

### Separation: is the confidence score measuring defects or measuring domain?

| population | AUC (max box confidence) | 95% CI | positives | negatives |
|---|---|---|---|---|
| cross-domain (Severstal, defective vs verified clean) | 0.567 | [0.545, 0.593] | 546 | 3336 |
| in-domain (NEU-DET test, defect vs mined clean) | 0.964 | [0.941, 0.983] | 197 | 771 |

The two mined arms are not balanced across crop sizes -- containment needs
room, so large crops are over-represented among the positives and small
crops among the negatives. Stratifying by crop size and recombining with
Mann-Whitney weights removes that composition effect:

| crop px | positives | negatives | AUC |
|---|---|---|---|
| 60 | 40 | 448 | 0.932 |
| 80 | 52 | 212 | 0.948 |
| 100 | 63 | 77 | 0.965 |
| 120 | 42 | 34 | 0.959 |

Size-stratified in-domain AUC: **0.943** (pooled 0.964). Either way the in-domain figure sits far above the cross-domain one, and the gap is not a crop-size artefact.

The in-domain negatives are mined crops that clear every labelled box by 6 px at sizes [60, 80, 100, 120] px -- NEU-DET has no defect-free image, so this proxy is the best available and it
flatters the model. Magnification is held equal across the two rows (1.28x); field of view is not, and cannot be, because a 200x200 NEU-DET frame has no room for a 200 px defect-free window.

### Shipped disposition chain on a clean coil -- 200x200 tiles (deployment geometry)

```
coil SEVERSTAL-CLEAN-HOLDOUT-AUDIT_NEGATIVES_ONLY: 480 frames, defect rate 0.4%, 3 detections, p95 severity 0.0 -> ACCEPT
  - Defect rate 0.4% within the accept limit of 2.0%.
  - 95th-percentile severity 0.0 within the accept limit of 30.0.
  - No frames in the high or critical severity band(s).
```

### Shipped disposition chain on a clean coil -- whole 1600x256 strip frames

```
coil SEVERSTAL-CLEAN-HOLDOUT-AUDIT_NEGATIVES_ONLY-WHOLEFRAME: 60 frames, defect rate 1.7%, 1 detections, p95 severity 0.0 -> ACCEPT
  - Defect rate 1.7% within the accept limit of 2.0%.
  - 95th-percentile severity 0.0 within the accept limit of 30.0.
  - No frames in the high or critical severity band(s).
```

Every frame on that coil is verified defect-free steel, so every detection
is a false alarm and every trigger listed above fired on nothing.

---

![clean false alarm rate vs defect recall](cross_domain_fa_vs_recall.png)

## How to read this, and what it does not say

- Severstal is hot-rolled carbon strip on someone else's line. A good score here
  would not prove the model works at Hisar. A bad score does prove it does not
  transfer, which is the asymmetry that makes the experiment worth running.
- Tiles from defective frames that carry no labelled pixels are excluded from both
  arms. They are not verified clean the way the clean arm is -- nobody annotated
  that region, on a frame that does carry a defect elsewhere -- and counting them
  as negatives would relabel the hardest cases in the set. Their flag rate is
  recorded per threshold in the JSON as `unlabelled_region_flag_rate`.
- Recall here is class-agnostic and crop-level: did any box appear on a tile that
  contains a labelled defect. It is not mAP and it is not comparable to the
  NEU-DET headline. Severstal's four classes do not map onto NEU-DET's six, so a
  class-aware score across the two datasets would be meaningless.
- The AUC is computed on max-box-confidence per tile. Tiles with no box score 0,
  which is a large tie mass; ranks are tie-averaged so the value does not depend
  on array order.

## Reproduce

```bash
.venv/bin/python src/cross_domain.py --tag baseline
```

Add `--weights <ckpt> --tag <name>` to score a new checkpoint into the same
JSON; every tagged run is redrawn on the chart, so a fix and its baseline sit on
one pair of axes.
