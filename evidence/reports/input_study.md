# Input pipeline: preserve the pixels, or interpolate them?

Generated 2026-09-09T16:48:08+00:00 by `src/input_study.py` on Apple M5 (mps), ultralytics 8.4.144, torch 2.14.0.

NEU-DET is natively 200x200. Every network input size that YOLO will accept is a multiple of 32, so every one of them resamples the source -- 224 is a 1.12x upscale, 256 a 1.28x upscale, 192 a 0.96x downscale that discards real pixels. But a 200x200 image fits inside a 224x224 canvas with 12 px of border on each side and **no scaling at all**, and inside 256x256 with 28 px. This study measures whether that lossless path is worth anything.

## Answer

**Scaled, not padded -- the lossless path loses, on these weights.** Padding a 200x200 source into a 224x224 or 256x256 canvas with no scaling does preserve every pixel byte-exact; that is verified below, and it is the first time this repository has had a genuinely lossless input path. It is also worse: it costs 0.024 to 0.052 mAP50 on the held-out test split, and it loses on every checkpoint, every split and both canvas sizes -- 8 of 8 configurations. Interpolation is not what limits this detector; the scale it expects objects at is. **This is a result about inference-time substitution only** -- see the confound stated in experiment 1.

**256 px, and the existing default is confirmed rather than replaced.** On the split a size may legitimately be chosen on, no input size in the 224-320 px band is distinguishable from 256 px on either checkpoint: the paired bootstrap (1000 resamples of 180 images, seed 1337) puts zero inside every one of those intervals. The val/test peak disagreement that prompted this study -- val peaking at 224 while test peaks at 256 -- is two noisy estimates of the same flat curve disagreeing, not a sign that the wrong size was chosen. **Keep `inference.DEFAULT_IMGSZ = 256`.**

**What actually matters in this knob is staying off the ends of the grid.** Inside the 224-320 px band every size is within 0.0254 mAP50 of its curve's best; at the ends of the grid the same curves give up as much as 0.1075, and the previously measured 640 px collapse (`reports/model_study.json`) is larger again. The failure mode here is inheriting the framework's 640 px default, not choosing 256 over 288.

| checkpoint | recommended input | test mAP50 | why |
|---|---|---|---|
| `yolov8n_neudet` | **256 px, scaled** (the ultralytics default path) | 0.7524 | val peaks at 224 px (0.7576 against 0.7467), but the paired val interval on that difference contains zero: 99.3% [-0.0142, +0.0396] (nominal 95% [-0.0088, +0.0319]), corrected for the peak having been selected from 8 sizes. No evidence to move off the incumbent |
| `yolov8n_joint` | **256 px, scaled** (the ultralytics default path) | 0.7642 | val peaks at 224 px (0.7835 against 0.7637), but the paired val interval on that difference contains zero: 99.3% [-0.0057, +0.0479] (nominal 95% [+0.0004, +0.0403]), corrected for the peak having been selected from 8 sizes. No evidence to move off the incumbent |

## What was actually fed to the network

Two things had to be established before any accuracy number meant anything, and both are measured rather than assumed.

**1. `imgsz` is not the network input size.** The ultralytics validator runs with `rect=True` and `pad=0.5`, and `BaseDataset.set_rectangle` computes `ceil(imgsz/32 + 0.5) * 32`. For a square image and an `imgsz` that is already a multiple of 32 that rounds up a whole stride, so **the tensor is `imgsz + 32`**. Validating at 256 px feeds a 288x288 tensor holding a 256x256 resized image inside a 16 px grey border. The framework was already padding; it just insisted on scaling first. Every published NEU-DET number in this repository should be read that way.

This is convenient rather than awkward: it makes `scale N` and `pad N` a matched pair. Same tensor, same compute, same border colour -- the only difference is whether the 200x200 source was interpolated up to N first.

**2. The padded path really is lossless, and the scaled path really is not.** The check pulls tensors out of the dataset object `model.val()` itself builds and compares them with the decoded source arrays:

| pipeline | imgsz | network tensor | images checked | result |
|---|---|---|---|---|
| pad | 224 | 256x256 | 180 | centre 200x200 byte-exact for 180/180, border value [114] |
| pad | 256 | 288x288 | 180 | centre 200x200 byte-exact for 180/180, border value [114] |
| scale | 224 | 256x256 | 180 | content is a bilinear resample for 180/180; resampling it back to 200x200 leaves errors up to 77 grey levels |
| scale | 256 | 288x288 | 180 | content is a bilinear resample for 180/180; resampling it back to 200x200 leaves errors up to 67 grey levels |

**3. This is the same instrument that produced the existing numbers.** 10 of the scaled cells below also appear in `reports/resolution_study.json`, measured before this study existed. The largest disagreement between the two is 0.00e+00 mAP50, which is exactly what a deterministic validator on identical inputs should give. The pad-versus-scale comparison is therefore not being made against a re-implemented baseline.

The padded copies were written as PNG (JPEG would re-quantise the pixels the exercise exists to preserve). Across all 720 written files, 720 survived the write-and-read-back round trip byte-exact, and the largest box round-trip error was 1.00e-08 source pixels -- boxes are shifted by an integer offset and never multiplied by a scale factor.

For completeness, what the resize costs in the image domain alone, with no model involved: resample the source to `imgsz` bilinearly, resample straight back to 200x200, and compare. Read the *categories*, not the ranking -- the residual for an upscale is mostly the return trip's own resampling, so these numbers do not say that 384 is more faithful than 224. What they do say is the qualitative thing that matters: 160 and 192 are downscales and destroy source pixels irrecoverably, everything from 224 up invents pixels rather than losing them, and the padded path is the only exactly-zero row.

| imgsz | scale factor | direction | round-trip MAE (grey levels) | PSNR (dB) |
|---|---|---|---|---|
| 160 | 0.8 | downscale | 2.9364 | 38.77 |
| 192 | 0.96 | downscale | 2.6076 | 39.81 |
| 224 | 1.12 | upscale | 2.4219 | 40.45 |
| 256 | 1.28 | upscale | 2.2997 | 40.9 |
| 288 | 1.44 | upscale | 2.2161 | 41.22 |
| 320 | 1.6 | upscale | 2.1472 | 41.49 |
| 352 | 1.76 | upscale | 2.1089 | 41.65 |
| 384 | 1.92 | upscale | 2.0754 | 41.79 |
| pad (any) | 1.0 | none (padded) | 0.0 | inf (exact) |

## Experiment 1 - pad versus scale

Both checkpoints, both splits. `scale 224` is the framework default path; `pad 224` and `pad 256` place the untouched 200x200 source in a grey canvas. All numbers are the ultralytics validator on its own defaults (conf 0.001, NMS IoU 0.7, max_det 300).

**val split**

| checkpoint | pipeline | network px | mAP50 | mAP50-95 | P | R | mAP50 vs scale at same imgsz |
|---|---|---|---|---|---|---|---|
| yolov8n_neudet | scale 224 | 256 | 0.7576 | 0.4291 | 0.659 | 0.743 | - |
| yolov8n_neudet | pad 224 | 256 | 0.7276 | 0.4127 | 0.623 | 0.722 | -0.0299 |
| yolov8n_neudet | scale 256 | 288 | 0.7467 | 0.4379 | 0.691 | 0.687 | - |
| yolov8n_neudet | pad 256 | 288 | 0.7390 | 0.4173 | 0.622 | 0.730 | -0.0076 |
| yolov8n_joint | scale 224 | 256 | 0.7835 | 0.4614 | 0.677 | 0.736 | - |
| yolov8n_joint | pad 224 | 256 | 0.7500 | 0.4275 | 0.727 | 0.652 | -0.0335 |
| yolov8n_joint | scale 256 | 288 | 0.7637 | 0.4497 | 0.668 | 0.722 | - |
| yolov8n_joint | pad 256 | 288 | 0.7497 | 0.4276 | 0.745 | 0.651 | -0.0140 |

**test split**

| checkpoint | pipeline | network px | mAP50 | mAP50-95 | P | R | mAP50 vs scale at same imgsz |
|---|---|---|---|---|---|---|---|
| yolov8n_neudet | scale 224 | 256 | 0.7270 | 0.3949 | 0.717 | 0.658 | - |
| yolov8n_neudet | pad 224 | 256 | 0.7032 | 0.3784 | 0.668 | 0.630 | -0.0238 |
| yolov8n_neudet | scale 256 | 288 | 0.7524 | 0.3967 | 0.696 | 0.687 | - |
| yolov8n_neudet | pad 256 | 288 | 0.7091 | 0.3874 | 0.655 | 0.649 | -0.0434 |
| yolov8n_joint | scale 224 | 256 | 0.7653 | 0.4125 | 0.757 | 0.703 | - |
| yolov8n_joint | pad 224 | 256 | 0.7132 | 0.3697 | 0.695 | 0.635 | -0.0521 |
| yolov8n_joint | scale 256 | 288 | 0.7642 | 0.4008 | 0.695 | 0.705 | - |
| yolov8n_joint | pad 256 | 288 | 0.7164 | 0.3695 | 0.686 | 0.655 | -0.0477 |

**Per-class AP50 on the held-out test split.**

*yolov8n_neudet*

| pipeline | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| scale 224 | 0.400 | 0.819 | 0.943 | 0.740 | 0.619 | 0.842 |
| pad 224 | 0.363 | 0.788 | 0.932 | 0.713 | 0.607 | 0.816 |
| scale 256 | 0.444 | 0.827 | 0.949 | 0.756 | 0.630 | 0.909 |
| pad 256 | 0.366 | 0.805 | 0.931 | 0.741 | 0.614 | 0.798 |

*yolov8n_joint*

| pipeline | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| scale 224 | 0.522 | 0.780 | 0.950 | 0.836 | 0.640 | 0.864 |
| pad 224 | 0.441 | 0.768 | 0.946 | 0.767 | 0.598 | 0.760 |
| scale 256 | 0.585 | 0.787 | 0.924 | 0.814 | 0.603 | 0.871 |
| pad 256 | 0.435 | 0.774 | 0.918 | 0.793 | 0.573 | 0.805 |

**Verdict.**

Preserving the pixels **loses** on every checkpoint, every split and both canvas sizes. On test the padded path costs between 0.0238 and 0.0521 mAP50 against the scaled path at the same tensor size. Interpolating the source, on these weights, beats keeping it byte-exact.

**This is not evidence that padding is a worse idea. It is evidence about inference-time substitution.** Both checkpoints were fine-tuned on scaled inputs and have never seen a centred 200x200 island of image inside a grey field, so a padded evaluation asks them to work on a geometry they were not optimised for, at an effective object scale they were not optimised for either. The confound is smaller than it looks -- `models/*/args.yaml` records `mosaic: 1.0`, `scale: 0.4`, `translate: 0.1`, `degrees: 10.0`, and ultralytics fills mosaic canvases and warp borders with the same value 114, so grey abutting steel is not novel to these networks -- but it is real, and the effective-scale change is not covered by it at all. Whether padding wins when the model is *trained* that way is a different experiment: fine-tune on `data/input_study/pad224` and re-run this table.

**Where it loses is consistent with that explanation rather than with a resampling artefact.** Per class on test, the padded path gives up most on the thin-structure classes and least on the large area-like ones -- `yolov8n_neudet` at pad256: scratches -0.111, crazing -0.078, ... pitted_surface -0.015; `yolov8n_joint` at pad256: crazing -0.150, scratches -0.067, ... patches -0.006. Padding shrinks every defect relative to the network's receptive field by the same factor the scaling would have grown it, and it is the fine-structure classes that cannot afford that. An interpolation artefact would not sort this way.

## Experiment 2 - the resolution grid, holes filled

160/192/224/256/288/320/352/384 on both splits for both checkpoints. The previously published grid skipped 288, 352 and 384, and its val and test peaks disagreed, which is exactly the situation in which a missing point can move the answer.

**val split, mAP50**

| checkpoint | 160 | 192 | 224 | 256 | 288 | 320 | 352 | 384 |
|---|---|---|---|---|---|---|---|---|
| yolov8n_neudet | 0.6873 | 0.7370 | 0.7576 | 0.7467 | 0.7505 | 0.7396 | 0.7070 | 0.6523 |
| yolov8n_joint | 0.7293 | 0.7707 | 0.7835 | 0.7637 | 0.7706 | 0.7588 | 0.7466 | 0.6962 |

**val split, mAP50-95**

| checkpoint | 160 | 192 | 224 | 256 | 288 | 320 | 352 | 384 |
|---|---|---|---|---|---|---|---|---|
| yolov8n_neudet | 0.3717 | 0.4144 | 0.4291 | 0.4379 | 0.4355 | 0.4315 | 0.4012 | 0.3419 |
| yolov8n_joint | 0.4074 | 0.4437 | 0.4614 | 0.4497 | 0.4550 | 0.4483 | 0.4221 | 0.3789 |

**test split, mAP50**

| checkpoint | 160 | 192 | 224 | 256 | 288 | 320 | 352 | 384 |
|---|---|---|---|---|---|---|---|---|
| yolov8n_neudet | 0.6589 | 0.7218 | 0.7270 | 0.7524 | 0.7423 | 0.7286 | 0.7032 | 0.6568 |
| yolov8n_joint | 0.6848 | 0.7260 | 0.7653 | 0.7642 | 0.7524 | 0.7454 | 0.7104 | 0.6578 |

**test split, mAP50-95**

| checkpoint | 160 | 192 | 224 | 256 | 288 | 320 | 352 | 384 |
|---|---|---|---|---|---|---|---|---|
| yolov8n_neudet | 0.3277 | 0.3752 | 0.3949 | 0.3967 | 0.3915 | 0.3926 | 0.3653 | 0.3244 |
| yolov8n_joint | 0.3356 | 0.3879 | 0.4125 | 0.4008 | 0.4010 | 0.3956 | 0.3627 | 0.3336 |

![resolution grid](input_study_grid.png)

**Peaks.**

| checkpoint | val peak | val mAP50 | test peak | test mAP50 | test mAP50 at 256 |
|---|---|---|---|---|---|
| yolov8n_neudet | 224 px | 0.7576 | 256 px | 0.7524 | 0.7524 |
| yolov8n_joint | 224 px | 0.7835 | 224 px | 0.7653 | 0.7642 |

**Verdict.**

`yolov8n_neudet`: val peaks at **224 px** (0.7576), test peaks at **256 px** (0.7524). Choosing on val and then reading test gives 0.7270; the shipped 256 px gives 0.7524; picking the size that maximises test would give 0.7524, which is **+0.0254** over the honest choice and is not available to anyone who has not already looked at the answer. That last number is the size of the cheat, not a result.

`yolov8n_joint`: val peaks at **224 px** (0.7835), test peaks at **224 px** (0.7653). Choosing on val and then reading test gives 0.7653; the shipped 256 px gives 0.7642; picking the size that maximises test would give 0.7653, which is **+0.0000** over the honest choice and is not available to anyone who has not already looked at the answer. That last number is the size of the cheat, not a result.

Across both checkpoints and both splits, every size from 224 to 320 px sits within 0.0254 mAP50 of that curve's best, while the ends of the grid fall away sharply (the grid ends give up to 0.1075 mAP50 against the same curve's best). The curve has an interior maximum on all four (checkpoint, split) combinations, so the grid brackets its own peak and is not reporting a boundary.

## Experiment 3 - is any of this real?

1000 bootstrap resamples of the 180 images in a split, seed 1337, **paired**: every configuration is scored on the same resampled image sets inside one loop, so the between-image variance they all share cancels in the difference and any two configurations can be compared consistently afterwards. Intervals are 2.5/97.5 percentiles of the paired difference against the shipped 256 px default.

Both splits are bootstrapped, and they do different jobs. **val is the decision instrument** -- the only interval a size may legitimately be chosen with, and the one that answers whether there is any evidence to move off the shipped default. **test is the honest measurement** of what a choice bought; selecting the size that minimises it would be cheating, and the two are shown side by side so that is visible rather than tempting.

Before any interval was reported, the mAP recomputed from the captured per-image rows was checked against the validator's own number for all 40 captured cells; the largest disagreement was 0.00e+00 mAP50.

**yolov8n_neudet, val split** (a size may be chosen on this) - mAP50 difference against `scale256`

| configuration | mAP50 | difference vs 256 | 95% CI | CI excludes 0 | wins in x% of resamples |
|---|---|---|---|---|---|
| pad224 | 0.7276 | -0.0190 | [-0.0428, +0.0052] | **no** | 5% |
| pad256 | 0.7390 | -0.0076 | [-0.0296, +0.0172] | **no** | 23% |
| scale160 | 0.6873 | -0.0594 | [-0.0908, -0.0262] | yes | 0% |
| scale192 | 0.7370 | -0.0096 | [-0.0324, +0.0129] | **no** | 22% |
| scale224 | 0.7576 | +0.0109 | [-0.0088, +0.0319] | **no** | 87% |
| scale288 | 0.7505 | +0.0038 | [-0.0145, +0.0208] | **no** | 65% |
| scale320 | 0.7396 | -0.0070 | [-0.0268, +0.0142] | **no** | 23% |
| scale352 | 0.7070 | -0.0397 | [-0.0683, -0.0093] | yes | 1% |
| scale384 | 0.6523 | -0.0943 | [-0.1264, -0.0532] | yes | 0% |

**yolov8n_joint, val split** (a size may be chosen on this) - mAP50 difference against `scale256`

| configuration | mAP50 | difference vs 256 | 95% CI | CI excludes 0 | wins in x% of resamples |
|---|---|---|---|---|---|
| pad224 | 0.7500 | -0.0136 | [-0.0366, +0.0111] | **no** | 15% |
| pad256 | 0.7497 | -0.0139 | [-0.0367, +0.0127] | **no** | 15% |
| scale160 | 0.7293 | -0.0343 | [-0.0677, +0.0039] | **no** | 4% |
| scale192 | 0.7707 | +0.0070 | [-0.0176, +0.0390] | **no** | 71% |
| scale224 | 0.7835 | +0.0198 | [+0.0004, +0.0403] | yes | 98% |
| scale288 | 0.7706 | +0.0069 | [-0.0101, +0.0272] | **no** | 78% |
| scale320 | 0.7588 | -0.0049 | [-0.0203, +0.0137] | **no** | 34% |
| scale352 | 0.7466 | -0.0170 | [-0.0401, +0.0082] | **no** | 9% |
| scale384 | 0.6962 | -0.0675 | [-0.0893, -0.0404] | yes | 0% |

**yolov8n_neudet, test split** (honest measurement; never selected on) - mAP50 difference against `scale256`

| configuration | mAP50 | difference vs 256 | 95% CI | CI excludes 0 | wins in x% of resamples |
|---|---|---|---|---|---|
| pad224 | 0.7032 | -0.0492 | [-0.0741, -0.0246] | yes | 0% |
| pad256 | 0.7091 | -0.0434 | [-0.0688, -0.0195] | yes | 0% |
| scale160 | 0.6589 | -0.0935 | [-0.1267, -0.0580] | yes | 0% |
| scale192 | 0.7218 | -0.0307 | [-0.0564, -0.0050] | yes | 1% |
| scale224 | 0.7270 | -0.0254 | [-0.0452, -0.0046] | yes | 1% |
| scale288 | 0.7423 | -0.0101 | [-0.0281, +0.0070] | **no** | 14% |
| scale320 | 0.7286 | -0.0238 | [-0.0470, +0.0019] | **no** | 3% |
| scale352 | 0.7032 | -0.0493 | [-0.0762, -0.0182] | yes | 0% |
| scale384 | 0.6568 | -0.0956 | [-0.1217, -0.0582] | yes | 0% |

**yolov8n_joint, test split** (honest measurement; never selected on) - mAP50 difference against `scale256`

| configuration | mAP50 | difference vs 256 | 95% CI | CI excludes 0 | wins in x% of resamples |
|---|---|---|---|---|---|
| pad224 | 0.7132 | -0.0509 | [-0.0762, -0.0227] | yes | 0% |
| pad256 | 0.7164 | -0.0477 | [-0.0733, -0.0175] | yes | 0% |
| scale160 | 0.6848 | -0.0794 | [-0.1104, -0.0415] | yes | 0% |
| scale192 | 0.7260 | -0.0382 | [-0.0607, -0.0114] | yes | 0% |
| scale224 | 0.7653 | +0.0011 | [-0.0189, +0.0221] | **no** | 56% |
| scale288 | 0.7524 | -0.0118 | [-0.0365, +0.0124] | **no** | 19% |
| scale320 | 0.7454 | -0.0188 | [-0.0466, +0.0083] | **no** | 11% |
| scale352 | 0.7104 | -0.0538 | [-0.0878, -0.0202] | yes | 0% |
| scale384 | 0.6578 | -0.1063 | [-0.1403, -0.0636] | yes | 0% |

![bootstrap intervals](input_study_bootstrap.png)

**Verdict.**

**val:** yolov8n_joint/scale224 beat 256 px at a nominal 95% level. That is a *selected maximum*, though, so the nominal level overstates the evidence: yolov8n_joint/scale224 becomes 99.3% [-0.0057, +0.0479], which contains zero once corrected for having been selected from 8 sizes. The recommendation uses the corrected interval, which is why it does not move off the incumbent.

**test:** in the 224-320 px band, **no size beats 256 px** on either checkpoint -- every interval that excludes zero in that band does so on the losing side. The intervals are wide, up to 0.0549 mAP50 across, against point differences of at most 0.0254, so with 180 images this benchmark mostly cannot resolve 224 from 320 px. The exceptions, and they cut against the smaller sizes rather than for them: yolov8n_neudet/scale224 ([-0.0452, -0.0046]).

18 of the 36 contrasts across both splits are resolved at all, and they are the ends of the grid and the padded configurations -- the differences big enough for 180 images to see.

The pad-versus-scale contrast at matched tensor size is the cleanest comparison in the study, because nothing but the resampling differs: `yolov8n_neudet` val pad224 - scale224: -0.0300 mAP50, CI [-0.0508, -0.0095] (resolved); `yolov8n_neudet` val pad256 - scale256: -0.0076 mAP50, CI [-0.0296, +0.0172] (contains zero); `yolov8n_joint` val pad224 - scale224: -0.0335 mAP50, CI [-0.0567, -0.0081] (resolved); `yolov8n_joint` val pad256 - scale256: -0.0139 mAP50, CI [-0.0367, +0.0127] (contains zero); `yolov8n_neudet` test pad224 - scale224: -0.0238 mAP50, CI [-0.0515, +0.0018] (contains zero); `yolov8n_neudet` test pad256 - scale256: -0.0434 mAP50, CI [-0.0688, -0.0195] (resolved); `yolov8n_joint` test pad224 - scale224: -0.0520 mAP50, CI [-0.0804, -0.0243] (resolved); `yolov8n_joint` test pad256 - scale256: -0.0477 mAP50, CI [-0.0733, -0.0175] (resolved).

## What this does not show

**Experiment 1 measures an inference-time substitution, not the value of padding.** Both checkpoints were trained on scaled inputs. A model fine-tuned on padded canvases could rank the two pipelines the other way; `data/input_study/pad224` exists so that experiment can be run without rebuilding anything.

**Everything is one pass per configuration.** Validation is deterministic given the weights and the input, so repeats would reproduce exactly; the variability quantified here is over *images*, not over runs, and not over training seeds. Separating training-seed variance would need several seeds per configuration.

**180 test images is a small benchmark.** That is precisely why experiment 3 exists, and why it declines to name a winner inside the plateau.

**Validation geometry is not deployment geometry.** `model.val()` pads square inputs to `imgsz + 32`; `model.predict()` on a single 200x200 frame letterboxes to `imgsz` with `auto=True` and no border at all. The accuracy ordering measured here is expected to carry over, but the two paths are not byte-identical and the deployed path was not separately swept.

**Severstal classes are excluded by construction.** The 10-class checkpoint is scored on NEU-DET labels, and ultralytics averages AP over classes present in the ground truth, so its four Severstal heads contribute nothing to these means. Predictions they emit on NEU-DET frames are invisible to mAP but would be visible to an operator; that is `src/false_alarm.py`'s question, not this one.

## Reproducing

```
.venv/bin/python src/input_study.py --resamples 1000
```

Wall clock 421 s for 40 validation passes plus 279.9 s of bootstrap. Raw numbers: `reports/input_study.json`.
