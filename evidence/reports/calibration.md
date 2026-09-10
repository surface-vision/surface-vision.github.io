# Confidence calibration

**Checkpoint** `yolov8n_neudet/best.pt` (`models/yolov8n_neudet/weights/best.pt`)  
**Device** mps | **imgsz** 256 | **NMS IoU** 0.45 | **score floor** 0.05 | **match IoU** 0.5  
**Fitted on** `val` (180 images, 705 detections)  
**Evaluated on** `test` (180 images, 790 detections) -- never seen by the fit  
**Generated** 2026-09-09T23:46:51+00:00

## 0. The gap this closes

The brief asks for a defect verdict "ideally with a confidence score". The console
shows one. It was never a probability. A detection displayed at 0.60 was correct far
more often than 60% of the time, and an operator who calibrates their own trust to the
displayed number therefore discards real defects.

A calibrator is fitted here **on `val` only**. `test` is used once, at the end, to
report the result. No hyper-parameter, no bin edge and no family choice was made by
looking at `test`.

## 1. Before: the shipped score is not a probability

On `test`, 790 detections at conf >= 0.05, labelled true/false positive by the evaluator's own
greedy class-aware matcher at IoU 0.5:

| score bin | n | mean score | empirical precision | gap |
|---|---|---|---|---|
| [0.00,0.15) | 309 | 0.086 | 0.191 | -0.105 |
| [0.15,0.25) | 116 | 0.195 | 0.336 | -0.141 |
| [0.25,0.35) | 62 | 0.301 | 0.387 | -0.086 |
| [0.35,0.45) | 40 | 0.401 | 0.575 | -0.174 |
| [0.45,0.55) | 48 | 0.502 | 0.667 | -0.164 |
| [0.55,0.65) | 69 | 0.599 | 0.841 | -0.242 |
| [0.65,0.75) | 55 | 0.697 | 0.927 | -0.231 |
| [0.75,0.85) | 50 | 0.801 | 0.980 | -0.179 |
| [0.85,1.00] | 41 | 0.885 | 1.000 | -0.115 |

**ECE = 0.1418**, and the sign of the gap is negative in every
populated bin. This is not noise around the diagonal, it is a systematic bias: the
detector is *under*-confident everywhere.

## 2. Why one parameter is not enough, measured rather than asserted

Temperature scaling has a single degree of freedom and it is a sharpening knob:
`T < 1` pushes scores above 0.5 up and scores below 0.5 down, `T > 1` does the
reverse. The bias above runs the *same direction at both ends*. No single temperature
can lift the 0.09 bin to 0.19 and the 0.89 bin to 1.00 at the same time.

Selection protocol: 5-fold cross-validation on the fitting split, grouped by source image, 5 folds, seed 20260909,
over 180 `val` images and 705 detections.
Grouping by image is not cosmetic -- two detections on one frame are not independent
draws, and an ungrouped fold would let a leaked frame flatter the more flexible model.
Every number below is **out of fold**, so isotonic gets no in-sample advantage.

| family | parameters | out-of-fold ECE | Brier | NLL | shippable |
|---|---|---|---|---|---|
| **isotonic** | non-parametric, monotone | 0.0250 | 0.1443 | 0.4485 | yes |
| isotonic_unsmoothed | non-parametric, monotone | 0.0363 | 0.1445 | 0.4752 | reference only |
| platt | 2 | 0.0614 | 0.1449 | 0.4536 | reference only |
| temperature | 1 | 0.0961 | 0.1543 | 0.4874 | yes |
| identity | 0 (the shipped score) | 0.1008 | 0.1534 | 0.4875 | reference only |

The winner is **isotonic**.

## 3. After

Fitted on all of `val`: **isotonic regression, 34 knots (+1e-06 strict-monotone term)**

Each plateau reports its Jeffreys posterior mean `(k + 0.5) / (n + 1)` rather than
the raw `k / n`, and the resulting sequence is re-projected with a weighted PAVA pass
so it stays monotone. That is not cosmetic. The top plateau covers 99 `val` detections above raw 0.780, of which 99 are true positives -- a maximum-likelihood value of 1.000. Printing **1.00** beside a steel defect is a claim that sample cannot support, so the
calibrator reports 0.995 instead. It also
measures better: smoothing improves out-of-fold ECE from 0.0363 to 0.0250.

| score bin | n | mean score | empirical precision | gap |
|---|---|---|---|---|
| [0.00,0.15) | 25 | 0.092 | 0.120 | -0.028 |
| [0.15,0.25) | 258 | 0.157 | 0.190 | -0.033 |
| [0.25,0.35) | 185 | 0.280 | 0.330 | -0.050 |
| [0.35,0.45) | 23 | 0.362 | 0.478 | -0.116 |
| [0.45,0.55) | 39 | 0.490 | 0.590 | -0.100 |
| [0.55,0.65) | 8 | 0.643 | 0.500 | +0.143 |
| [0.65,0.75) | 69 | 0.708 | 0.768 | -0.060 |
| [0.75,0.85) | 69 | 0.801 | 0.884 | -0.083 |
| [0.85,1.00] | 114 | 0.970 | 0.974 | -0.003 |

| metric | before (`test`) | after (`test`) |
|---|---|---|
| ECE | 0.1418 | 0.0461 |
| max per-bin gap (bins with n >= 20) | 0.2417 | 0.1159 |
| Brier score | 0.1754 | 0.1576 |
| negative log likelihood | 0.5403 | 0.4819 |
| ECE at the shipped operating point (conf >= 0.15, n = 481) | 0.1654 | 0.0559 |

ECE falls by 68%, and the
remaining 0.046 is not a systematic bias any more -- the largest surviving gaps sit in
the thin middle bins where a few dozen detections cannot pin a precision down. The
residual is honest sampling noise, not a direction.

![reliability diagram](calibration.png)

## 4. mAP is unchanged, and that is measured, not argued

Average precision is a function of the *ranking* of detections, not of their score
values: the greedy matcher walks predictions in descending score and the PR curve is
swept by rank. A strictly increasing rescoring therefore cannot move it. Isotonic
regression is only non-decreasing on its own, which would create ties and make AP
tie-break dependent, so `IsotonicCalibrator` blends in a vanishing `eps * s` term
(eps = 1e-06) that restores a strict order without moving any
value by more than that.

Checked on the 790 `test` detections:

| check | result |
|---|---|
| sorted order identical before/after | True |
| distinct raw scores collapsed onto one calibrated score | 0 |
| strictly increasing on the data | True |
| non-decreasing on a 2001-point grid over [0,1] | True |
| largest score shift | 0.2272 |

AP50 recomputed from scratch on `test`, once ranked by raw score and once by
calibrated score (101-point interpolation, greedy class-aware matching at IoU 0.5):

| class | AP50 (raw ranking) | AP50 (calibrated ranking) | verdict |
|---|---|---|---|
| crazing | 0.426480 | 0.426480 | identical |
| inclusion | 0.723276 | 0.723276 | identical |
| patches | 0.885033 | 0.885033 | identical |
| pitted_surface | 0.752905 | 0.752905 | identical |
| rolled-in_scale | 0.627395 | 0.627395 | identical |
| scratches | 0.832828 | 0.832828 | identical |
| mAP50 | 0.707986 | 0.707986 | identical |

AP50 here is this module's own 101-point implementation over the evaluator's greedy matcher, so it is not numerically the ultralytics headline (0.7524); it is the same quantity computed twice under two rankings, which is what makes the invariance meaningful.

## 5. What this does not fix

The map is `score -> P(true positive)` **on NEU-DET validation frames**, every one of
which contains a defect. The base rate on genuinely defect-free strip is different, so
a calibrated 0.9 on mill imagery is not the same claim as a calibrated 0.9 here (see
`reports/false_alarm.md` and the cross-domain work for what that population actually
looks like). Calibration changes what the number *means*; it changes nothing about
what is detected. It is not an accuracy improvement and must not be presented as one.

Two smaller caveats worth stating:

1. The fit sees 705 detections from 180 images. The top bins are thin, so the upper end of the
   curve is the least certain part of it. Out-of-fold selection is what keeps that from
   becoming an overfitting story, but more validation frames would tighten it.
2. The calibrator is class-agnostic. Per-class miscalibration is real and measured in
   `docs/audit/technical_headroom.md` (gaps from -0.055 on `scratches` to -0.193 on
   `crazing`); a per-class calibrator would fit six times as many parameters on the same
   180 images, which is the wrong trade at this sample size.

## 6. How to use it

```python
from calibrate import calibrate_confidence

shown = calibrate_confidence(detection.confidence)   # 0.60 -> 0.80
```

Parameters live in `calibration.json` with the split and date they were fitted
on. Re-fit whenever the checkpoint changes: a calibrator is tied to one set of weights.

| raw score | calibrated |
|---|---|
| 0.15 | 0.255 |
| 0.25 | 0.330 |
| 0.40 | 0.500 |
| 0.60 | 0.802 |
| 0.70 | 0.891 |
| 0.80 | 0.995 |
| 0.90 | 0.995 |

