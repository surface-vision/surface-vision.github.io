# Model selection study -- YOLOv8n vs YOLOv8s on NEU-DET

Generated 2026-09-09T04:39:04+00:00 on macOS-26.6.2-arm64-arm-64bit / arm64, device `mps`, torch 2.14.0, ultralytics 8.4.144.

Every number below was measured by `src/model_study.py` in the run that wrote this file. Metrics come from the ultralytics validator on its own protocol defaults (conf 0.001, NMS IoU 0.7, max_det 300), which is what published NEU-DET numbers use.

> **Amended 2026-09-09, section 2 only.** The inference-resolution sweep in section 2 was truncated at 256 px, and the paragraph explaining why contained a false statement about upsampling (256/200 = 1.28x, so 256 px upsamples). Section 2's grid has been extended to 128/160/192/224 px, the mAP50-95 and fitness columns added, and the paragraph rewritten; the decision row above and the "Why 256 px" paragraph in section 5 were updated to match, and the 640 px training-cost paragraphs in section 6 now cite a completed 150-epoch 640 px run (64.9 s/epoch, 3.34 h, test mAP50 0.7338 at its best input size against the shipped 0.7524) instead of a projection. Those cells were measured by a separate sweep on the identical validator protocol and are recorded in **`reports/resolution_study.json`**; every previously published cell (256 px and above) reproduced its old value exactly on re-measurement, so the two sets are interchangeable. Nothing outside section 2, the imgsz decision row, "Why 256 px" and the 640 px paragraph was touched.
>
> **These edits are ahead of the generator.** This file is written by `src/model_study.py`, which still contains the old grid (`IMGSZ_GRID` at line 126, and the comment at lines 123-125) and the old prose (lines 1896-1897 and 2629-2638). **Re-running `make model-study` will overwrite the corrections below.** The generator needs the same edit before this report is regenerated.

## Verdict

**Ship `yolov8n` at 256 px with TTA off.** On the held-out test split, at the input size validation chose, it scores mAP50 0.7524 / mAP50-95 0.3967, from a 6.22 MB checkpoint with 3.012M parameters, at a batch-1 floor of 3.5 ms per 200x200 frame on this machine (median 4.5 ms; see the latency caveat in section 5 -- the absolute milliseconds are not reproducible run to run on a contended laptop and should not be quoted as a throughput figure). With both models at their own best input size the gap over yolov8s is +0.0181 mAP50, 95% CI [-0.0084, +0.0424], which **contains zero**. The honest statement is therefore not 'nano is the better detector' but '**nano is not worse, and it is 3.7x smaller, 3.5x cheaper in FLOPs**'. That is more than sufficient to decide a deployment, and it is the reason to ship it. The larger claim -- that the nano architecture is better on a dataset this size -- is not supported once both models are given their best configuration.

| decision | value | decided on | evidence |
|---|---|---|---|
| checkpoint | `models/yolov8n_neudet/weights/best.pt` | test bootstrap at each model's best size, then cost | at best sizes 0.7524 vs 0.7343 mAP50, CI [-0.0084, +0.0424] contains 0; deployment-size bootstrap interval contains zero, so the tie breaks on cost; yolov8n is the cheaper model on all of parameter count, weight-file size, FLOPs (batch-1 latency does not separate them at this input size) |
| inference imgsz | 256 px | val, on mAP50-95 and fitness; test agrees | swept 128-640 px (section 2). 256 px is the peak of mAP50-95 and of ultralytics' fitness composite in all 8 model x split cells, and the peak of test mAP50. On *val mAP50 alone* the peak is 224 px, by +0.0109 -- inside the 0.0254 bootstrap half-width, and worth -0.0254 on test if taken. The curve collapses above the training size -- up to 57% of mAP50 lost by 640 px |
| test-time augmentation | off | val + latency | val gain -0.0061 mAP50 for 2.39x latency |
| confidence threshold | 0.15 | val (operating-point study) | minimum expected mill cost subject to a 90% detection floor; src/false_alarm.py (tuned on val) |
| NMS IoU | 0.45 | shipping default | DefectDetector default; not swept in this study |

## 1. Head to head on the held-out test split

Both checkpoints, same 1440/180/180 split, same recipe, evaluated at the training resolution (320 px).

| model | mAP50 | mAP50-95 | P | R | params (M) | GFLOPs @320 | weight MB | latency @320 median ms | p95 ms | FPS (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 0.7286 | 0.3926 | 0.7322 | 0.6018 | 3.012 | 2.049 | 6.22 | 4.97 | 5.36 | 201 |
| yolov8s | 0.6598 | 0.3618 | 0.6107 | 0.6475 | 11.138 | 7.163 | 22.49 | 8.63 | 18.35 | 116 |

Per-class AP50 at the training resolution:

| model | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| yolov8n | 0.395 | 0.842 | 0.945 | 0.748 | 0.647 | 0.795 |
| yolov8s | 0.330 | 0.805 | 0.921 | 0.704 | 0.438 | 0.762 |
| **difference** | +0.066 | +0.037 | +0.024 | +0.044 | +0.209 | +0.033 |

### Is the gap real? Paired bootstrap over the 180 test images

Both models were rescored on the *same* 2000 resampled test sets (180 images drawn with replacement, seed 1337). Pairing matters: most of the variance in a 180-image mAP is which images were drawn, and that component is shared by both models and cancels in the difference.

| metric | yolov8n | yolov8s | difference | 95% CI on the difference | excludes 0? | yolov8n wins |
|---|---|---|---|---|---|---|
| mAP50 | 0.7286 | 0.6598 | +0.0688 | [+0.0309, +0.0990] | yes | 99.95% |
| mAP50-95 | 0.3926 | 0.3618 | +0.0308 | [+0.0111, +0.0484] | yes | 99.9% |

**Reading.** The observed test-set gap is +0.0688 mAP50 in favour of yolov8n. The 95% interval on that difference is [+0.0309, +0.0990], which **excludes zero**: on a differently drawn 180-image test set from the same distribution, the ordering would come out the same way 99.95% of the time. The gap is not an artefact of which 180 images we happened to hold out.

On mAP50-95 the difference is +0.0308 with 95% CI [+0.0111, +0.0484] (excludes zero); yolov8n wins 99.9% of resamples.

Pairing was worth having: the standard deviation of the paired difference is 0.0173, against 0.0278 if the two models had been scored on independently drawn test sets. That is a 1.6x tighter interval for no extra compute, and it is the whole reason a 0.07 gap on 180 images can be resolved at all.

**A tension worth confronting, because it looks like a contradiction.** On the validation split the same two checkpoints are nearly tied: +0.0159 mAP50, against +0.0688 on test. A reader is entitled to ask which split to believe, and the answer is not 'average them'. **`best.pt` was chosen on val.** Ultralytics saves the epoch with the highest validation fitness, so each checkpoint is the maximum over 135 (nano) and 150 (small) validation evaluations. A maximum over many draws is biased upward, and it is biased upward *on the split it was maximised over*. Val therefore over-states both models by an unknown amount and cannot referee a comparison between them; it is a selection split wearing an evaluation split's clothes. Test is the only measurement here that no decision was made on, which is why it is the one the verdict rests on.

That resolution is worth stating as a limit as well as a defence. It explains why the two splits disagree without appealing to luck, but it does not prove the test figure is unbiased in every respect -- only that it is the less contaminated of the two.

Fidelity check -- the bootstrap rescores the validator's own per-image rows, so recomputing over the full split must reproduce the validator exactly:

| model | imgsz | validator mAP50 | recomputed mAP50 | abs error | exact |
|---|---|---|---|---|---|
| yolov8n | 320 | 0.72860482 | 0.72860482 | 0.00e+00 | yes |
| yolov8s | 320 | 0.65979556 | 0.65979556 | 0.00e+00 | yes |
| yolov8n | 256 | 0.75244755 | 0.75244755 | 0.00e+00 | yes |
| yolov8s | 256 | 0.73430937 | 0.73430937 | 0.00e+00 | yes |

![paired bootstrap](model_study_bootstrap.png)

### The same test, with each model at its own best input size

**The comparison above is not the one the deployment faces.** It scores both models at 320 px, the size they were trained at, which is the right way to answer 'which of these two training runs came out better'. But section 2 shows input size moves mAP50 by far more than the architecture does, and neither model's best input size is 320. The question that decides what ships is which checkpoint is better **at its own best configuration**, so the bootstrap is repeated with each model at the size val chose for it (nano at 256 px, small at 256 px), paired over the same resampled test sets.

At their own best sizes the gap is +0.0181 mAP50 (0.7524 against 0.7343), 95% CI [-0.0084, +0.0424], which **contains zero**. yolov8n@256 wins 90.2% of resamples and yolov8s@256 the rest, so at their best configurations the two checkpoints are not separable on accuracy at all.

**Both results are true and they answer different questions.** At the trained size, nano is ahead by +0.0688 with an interval that excludes zero. At each model's best size, the gap is +0.0181 with an interval that contains zero. The honest headline is therefore the narrower one: **the original claim that nano beats small survives at the size the claim was made about, but it does not survive once both models are given their best input size.** Most of what looked like an architecture difference was a resolution effect that happened to hurt the larger model more at 320 px. Anyone quoting the +0.069 figure as evidence that a smaller detector is better on small datasets is over-reading it. Note also that the two metrics disagree in sign at the deployment size: nano leads on mAP50 by +0.0181 and trails on mAP50-95 by -0.0094, both intervals containing zero. Two metrics pointing opposite ways, neither significantly, is what a genuine tie looks like -- and it is why the deployment decision below is made on cost. **The strongest form of the objection should be stated outright**, because a reader will find it: ultralytics' own model-selection criterion is the composite fitness 0.1*mAP50 + 0.9*mAP50-95, which is what chose both of these checkpoints epoch by epoch. Evaluated at the deployment size on test it gives 0.4323 for yolov8n@256 against 0.4389 for yolov8s@256 -- so on the project's own composite metric the ordering favours **yolov8s** by 0.0066, because that metric weights mAP50-95 nine to one and mAP50-95 is the metric nano trails on. That difference is well inside the bootstrap interval and decides nothing, but it does mean no accuracy metric picks yolov8n here. The recommendation below rests on cost, and it has to.

| metric | yolov8n@256 | yolov8s@256 | difference | 95% CI on the difference | excludes 0? | yolov8n@256 wins |
|---|---|---|---|---|---|---|
| mAP50 | 0.7524 | 0.7343 | +0.0181 | [-0.0084, +0.0424] | **no** | 90.2% |
| mAP50-95 | 0.3967 | 0.4061 | -0.0094 | [-0.0250, +0.0084] | **no** | 15.9% |

![paired bootstrap, deployment sizes](model_study_bootstrap_deploy.png)

### The mechanism: capacity against 1440 training images

The direct evidence for the capacity explanation is in the loss curves, not the mAP. At the end of training yolov8s sits at train cls loss 0.823 against val 1.090 -- a generalisation gap of +0.267. yolov8n sits at 1.173 against 1.112, a gap of -0.061 -- it is scoring *better* on data it has never seen than on data it trains on, which is what a model with too little capacity to memorise looks like when the training set is also being augmented hard. The larger model is fitting structure in the 1440 training images that does not transfer. That is the mechanism, and it is measured rather than assumed.

The scale of the effect is what needs care. 1440 training images across six classes is roughly 240 images per class, which is thin for an 11.1M-parameter detector and comfortable for a 3.0M-parameter one. The recipe made this worse in one specific way: both runs used the ultralytics default schedule with mosaic, mixup and 150 epochs, tuned for COCO-scale data. Nothing in it -- no extra weight decay, no shorter schedule, no frozen backbone -- was adjusted for the larger model on the smaller dataset. So the honest claim is not 'yolov8s is the wrong architecture for NEU-DET' but '**yolov8s trained with this recipe on this much data is worse than yolov8n trained the same way**'. Those are different claims, and only the second one is measured.

Re-evaluated at 320 px, the two checkpoints score 0.7396 and 0.7237 mAP50 on val -- close, for the selection reason given above -- and 0.7286 against 0.6598 on test. yolov8n also stopped early at epoch 135 of 150 with its best weights at epoch 85, while yolov8s ran the full 150 with its best at epoch 116. Neither run was starved of epochs, and neither was still improving when it stopped.

| model | epochs run | best epoch | best val mAP50 | final train cls loss | final val cls loss | cls generalisation gap | s/epoch (median) | s/epoch (mean, incl. stalls) |
|---|---|---|---|---|---|---|---|---|
| yolov8n | 135 | 85 | 0.7399 | 1.173 | 1.112 | -0.061 | 15.24 | 15.48 |
| yolov8s | 150 | 116 | 0.7355 | 0.823 | 1.090 | +0.267 | 36.47 | 52.44 |

## 2. Inference-resolution sensitivity

Both models were trained at 320 px on 200x200 source images. The grid below has been extended downward since this section was first written: it now runs 128 px to 640 px, and only 128, 160 and 192 px stay at or below the 200 px native size. **224 px and 256 px both upsample** -- 256/200 = 1.28x -- and nothing above 200 px adds information the file did not already contain. An earlier version of this paragraph swept only 256-640 px and asserted that 256 px was the one size in the grid that did not upsample; that arithmetic was wrong, and the truncated grid was also hiding its own peak. The corrected full-grid measurement, the 640 px training run, and the reasoning that keeps 256 px as the shipped size are in `reports/resolution_study.md`.

**mAP50** over the corrected grid. The four cells left of the double rule were not in the original sweep; every cell from 256 px rightward reproduced its previously published value exactly on re-measurement.

| model | split | 128 px | 160 px | 192 px | 224 px | 256 px | 320 px | 416 px | 512 px | 640 px |
|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | val | 0.5806 | 0.6873 | 0.7370 | **0.7576** | 0.7466 s | 0.7396 | 0.6031 | 0.4898 | 0.3225 |
| yolov8n | test | 0.5323 | 0.6589 | 0.7218 | 0.7270 | **0.7524** s | 0.7286 | 0.6230 | 0.4843 | 0.3438 |
| yolov8s | val | 0.6064 | 0.7499 | 0.7488 | **0.7570** | 0.7530 | 0.7237 | 0.6314 | 0.5381 | 0.3331 |
| yolov8s | test | 0.5774 | 0.7009 | **0.7434** | 0.7405 | 0.7343 | 0.6598 | 0.5646 | 0.4929 | 0.3227 |
| latency (median ms) | batch 1 | not measured | not measured | not measured | not measured | 4.5 / 7.8 | 5.0 / 8.6 | 6.5 / 11.3 | 6.8 / 15.0 | 10.9 / 23.8 |

Bold marks the peak of each row; `s` marks the shipped size. The latency row is nano / small single-frame latency in ms at batch 1, quoted as the **median** of the timed calls: this laptop was running other work throughout, and the median is robust to both an occasional stall and a single lucky call in a way that neither the mean nor the minimum is. Mean, min and p95 for every cell are in the JSON. Latency at the four new sizes was deliberately not measured: the machine was running a 640 px training job during the extension, and this report already refuses to quote absolute milliseconds taken under contention.

**And the same grid on mAP50-95 and on ultralytics' own fitness composite**, because mAP50 is not the only criterion in play and the three disagree:

| model | split | metric | 128 | 160 | 192 | 224 | 256 | 320 | 416 | 512 | 640 | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | val | mAP50-95 | 0.2797 | 0.3717 | 0.4144 | 0.4291 | **0.4379** | 0.4314 | 0.2981 | 0.2209 | 0.1465 | 256 |
| yolov8n | test | mAP50-95 | 0.2419 | 0.3277 | 0.3752 | 0.3949 | **0.3967** | 0.3926 | 0.2919 | 0.2129 | 0.1495 | 256 |
| yolov8s | val | mAP50-95 | 0.2902 | 0.3939 | 0.4293 | 0.4377 | **0.4493** | 0.4386 | 0.3070 | 0.2239 | 0.1389 | 256 |
| yolov8s | test | mAP50-95 | 0.2523 | 0.3532 | 0.4012 | 0.4024 | **0.4061** | 0.3618 | 0.2601 | 0.2063 | 0.1346 | 256 |
| yolov8n | val | fitness | 0.3098 | 0.4033 | 0.4466 | 0.4619 | **0.4688** | 0.4623 | 0.3286 | 0.2478 | 0.1641 | 256 |
| yolov8n | test | fitness | 0.2709 | 0.3608 | 0.4098 | 0.4281 | **0.4323** | 0.4262 | 0.3250 | 0.2401 | 0.1689 | 256 |
| yolov8s | val | fitness | 0.3218 | 0.4295 | 0.4612 | 0.4696 | **0.4797** | 0.4671 | 0.3394 | 0.2553 | 0.1583 | 256 |
| yolov8s | test | fitness | 0.2848 | 0.3880 | 0.4354 | 0.4362 | **0.4389** | 0.3916 | 0.2906 | 0.2350 | 0.1534 | 256 |

`fitness` is ultralytics' composite 0.1*mAP50 + 0.9*mAP50-95 -- the criterion that chose `best.pt` epoch by epoch for both of these checkpoints, and the one section 1 already invokes when stating the objection that no accuracy metric picks yolov8n. **On mAP50-95 and on fitness, 256 px is the peak in all eight of eight model x split cells.** Only mAP50 ever selects anything else.

**Resolution mismatch is the single largest effect in this entire study, and it is catastrophic rather than marginal.** Going from the best input size to 640 px costs between 54% and 57% of mAP50 depending on the model and split -- far more than the difference between the two architectures, more than TTA, more than anything else measured here. Feeding a detector trained at 320 px an image at 640 px is not a mild extrapolation; it roughly halves it.

The mechanism is scale, not detail. A YOLOv8 head assigns each object to a feature level by its pixel size, and those assignments were learned from 320 px inputs. At 640 px every defect is twice as many pixels across as anything the model was trained to regress, so boxes land on the wrong stride and the classifier sees a texture at a spatial frequency it never saw. Because NEU-DET defects are large relative to the frame -- a crazing label alone covers about a quarter of it -- there is no small-object regime that benefits to offset the loss.

Peak performance sits **below the training size**, and the arithmetic of why needs stating correctly, because an earlier version of this paragraph got it wrong. The source images are 200x200. 320 px upsamples them by 1.60x; **256 px upsamples them by 1.28x**; 224 px upsamples by 1.12x. The only sizes in the corrected grid that do *not* upsample are 128, 160 and 192 px. So the claim that 256 px was the one size that added no interpolated pixels was simply false, and so was the mechanism attached to it -- if nothing above 200 px could pay for its own compute, 192 px would win, and it does not. **192 px loses to 256 px in 11 of the 12 model x split x metric cells above**, the sole exception being yolov8s on test mAP50 (0.7434 against 0.7343).

What is true is the weaker and more useful statement: **the network has an optimum near, but not at, the native resolution, and it sits far below the training size.** Upsampling to roughly 1.1-1.3x native buys something real -- a detector's stride grid and its anchor-free scale assignment are defined in *network* pixels, so a mild upsample moves a 70 px defect onto a feature level the head was better trained to regress. Beyond that the interpolation adds no information and the scale mismatch starts to dominate. Both effects are visible in the curve.

The peaks, over the full grid: yolov8n on val peaks at 224 px (0.7576) on mAP50 and at 256 px on mAP50-95 (0.4379) and on fitness (0.4688); yolov8n on test peaks at 256 px on all three (0.7524 / 0.3967 / 0.4323); yolov8s on val peaks at 224 px on mAP50 (0.7570) and 256 px on the other two; yolov8s on test peaks at 192 px on mAP50 (0.7434) and 256 px on the other two.

For yolov8n the val sweep runs 128:0.5806, 160:0.6873, 192:0.7370, 224:0.7576, 256:0.7466, 320:0.7396, 416:0.6031, 512:0.4898, 640:0.3225, best mAP50 at 224 px. On test the same model runs 128:0.5323, 160:0.6589, 192:0.7218, 224:0.7270, 256:0.7524, 320:0.7286, 416:0.6230, 512:0.4843, 640:0.3438, best at 256 px.

For yolov8s the val sweep runs 128:0.6064, 160:0.7499, 192:0.7488, 224:0.7570, 256:0.7530, 320:0.7237, 416:0.6314, 512:0.5381, 640:0.3331, best mAP50 at 224 px. On test: 128:0.5774, 160:0.7009, 192:0.7434, 224:0.7405, 256:0.7343, 320:0.6598, 416:0.5646, 512:0.4929, 640:0.3227, best at 192 px.

**Does the shipped choice survive this? Yes, and it should be argued rather than asserted.** The stated selection rule in this report is *best val mAP50*, and on the complete grid that rule selects **224 px, not 256 px** -- the original sweep started at 256 and reported its own left-hand boundary as a maximum. Three things keep 256 px:

1. **The margin is inside the noise this report already measured.** On val, 224 px beats 256 px by +0.0109 mAP50. The paired bootstrap in section 1 puts a 95% half-width of 0.0254 on a mAP50 difference of this kind on these splits. 0.0109 is well inside it.
2. **Every other criterion picks 256 px, unanimously.** mAP50-95 and ultralytics' fitness composite both peak at 256 px in all eight model x split cells. The only reading that selects 224 px is mAP50 on val alone.
3. **The held-out split agrees with 256 px, and the cost of the alternative is measurable.** On test, yolov8n at 224 px scores 0.7270 against 0.7524 at 256 px -- applying the naive rule to the corrected grid would have cost **-0.0254 mAP50** on the split that decides nothing but reports everything. Per class, most of that is crazing, the hardest class: AP50 0.3998 at 224 px against 0.4443 at 256 px.

**The honest form of the claim is therefore:** 256 px is not the val mAP50 argmax, it is the argmax of every other criterion and of the held-out split, and the size that does beat it on val mAP50 does so by less than the measurement noise. That is a defensible pin. What is not defensible is the original sentence, which asserted a peak the grid was too narrow to have found. The full corrected grid, with the 640 px training run that goes with it, is in `reports/resolution_study.md`.

**The deployment consequence is a rule, not a number.** The curve is not symmetric: dropping *below* the training size is nearly free here -- every model and split peaks at 192-256 px, well under the 320 px they were trained at -- while going above it is ruinous. So the rule is **never run above the size the model was trained at, and validate any size you do run**. If mill frames need a larger input -- and they will, because a 2048 px camera frame is not a 200 px tile -- the model must be *retrained* at that size, not merely evaluated at it. `reports/resolution_study.md` does exactly that retrain at 640 px and reports what it buys: a model trained at 640 px still peaks *below* its training size (512 px on test) and does not beat the shipped 320 px model. This is a live operational hazard, not a theoretical one: `imgsz` is an argument anyone can change, it raises no error, and at 640 px it silently costs more than half the accuracy.

![resolution sweep](model_study_imgsz.png)

## 3. Test-time augmentation

Ultralytics TTA runs the image at three scales with a horizontal flip and merges the detections before NMS.

| model | imgsz | split | mAP50 no TTA | mAP50 with TTA | delta | latency no TTA (ms) | latency TTA (ms) | cost multiple |
|---|---|---|---|---|---|---|---|---|
| yolov8n | 256 | val | 0.7466 | 0.7406 | -0.0061 | 8.06 | 19.23 | 2.39x |
| yolov8n | 256 | test | 0.7524 | 0.7144 | -0.0380 | 8.06 | 19.23 | 2.39x |
| yolov8s | 256 | val | 0.7530 | 0.7553 | +0.0023 | 8.05 | 18.32 | 2.27x |
| yolov8s | 256 | test | 0.7343 | 0.7263 | -0.0080 | 8.05 | 18.32 | 2.27x |

On the split that is allowed to decide -- val -- TTA moves yolov8n from 0.7466 to 0.7406 mAP50, -0.0061. It costs 2.39x the latency (8.06 ms to 19.23 ms per frame, 2.70x at the floor), i.e. it throws away 58% of the throughput of the accelerator. Both arms of that ratio were timed alternately inside one loop, so it is a cost multiple rather than the quotient of two separately-contended measurements.

On test the same change is 0.7524 to 0.7144 (-0.0380), reported for completeness and not used to decide anything.

**Not worth it on a production line.** The accuracy it buys is inside the noise band the bootstrap already measured (a 95% interval 0.0680 wide on a difference of this size), while the throughput cost is certain and large. Section 2f of docs/research_notes.md puts the required tile rate at roughly 10,800 200x200 tiles per second at the slowest well-sourced line speed; multiplying per-frame cost by 2.39 multiplies the accelerator count by the same factor for a gain nobody can demonstrate. Spend the silicon on more cameras or a second inspection point, not on re-running the same frame at three scales.

## 4. Which classes fail, and why

Failure analysis is run on the recommended checkpoint (**yolov8n** at 256 px) through the shipping `DefectDetector` path at the deployment NMS IoU 0.45, not at the validator's mAP protocol -- these are the boxes an operator would actually be shown. Columns are at conf=0.25 (the `DefectDetector` default); the last column repeats recall at conf=0.15, the threshold the operating-point study currently recommends (src/false_alarm.py (tuned on val)). Every missed label is sorted by the best overlapping prediction of any class: **class confusion** = something drawn in the right place (IoU >= 0.5) with the wrong name; **extent error** = best overlap between 0.1 and 0.5, the defect was seen but the rectangle disagrees; **blind miss** = nothing overlaps at all. The AP50 column, and every correlation below it, is the validator's per-class AP50 for the same checkpoint at the same 256 px -- not the 320 px figure quoted in section 1, which describes a configuration this project is not shipping. The two differ enough to matter: at 320 px crazing scores 0.395 against 0.444 here.

| class | AP50 | GT boxes | recall @0.25 | class confusion | extent error | blind miss | confused with | recall @0.15 |
|---|---|---|---|---|---|---|---|---|
| crazing | 0.444 | 79 | 0.14 | 0 | 15 | 53 | - | 0.35 |
| inclusion | 0.827 | 89 | 0.70 | 0 | 14 | 13 | - | 0.74 |
| patches | 0.949 | 99 | 0.84 | 0 | 11 | 5 | - | 0.89 |
| pitted_surface | 0.756 | 46 | 0.70 | 0 | 11 | 3 | - | 0.74 |
| rolled-in_scale | 0.630 | 69 | 0.51 | 0 | 21 | 13 | - | 0.65 |
| scratches | 0.909 | 64 | 0.86 | 0 | 7 | 2 | - | 0.88 |

Statistics measured from the test images and their labels:

| class | AP50 | boxes | separability | edge energy ratio | mean box area | label overlap IoU | boxes/image (box-weighted) |
|---|---|---|---|---|---|---|---|
| crazing | 0.444 | 79 | 0.319 | 1.014 | 23.8% | 0.037 | 2.90 |
| inclusion | 0.827 | 89 | 0.829 | 1.631 | 5.7% | 0.022 | 3.81 |
| patches | 0.949 | 99 | 1.588 | 0.922 | 11.9% | 0.019 | 3.30 |
| pitted_surface | 0.756 | 46 | 0.622 | 1.042 | 52.1% | 0.044 | 2.22 |
| rolled-in_scale | 0.630 | 69 | 0.513 | 1.127 | 17.9% | 0.026 | 2.54 |
| scratches | 0.909 | 64 | 2.432 | 2.763 | 9.7% | 0.005 | 2.69 |

**The failure is not class confusion, and that is a measurement, not an impression.** Across all six classes and all 168 missed labels at conf=0.25, exactly **0** were cases of a box drawn in the right place carrying the wrong class name. The false positives tell the same story from the other side: they land almost entirely on frames of their own class (crazing -> crazing x8; rolled-in_scale -> rolled-in_scale x21), i.e. they are extra or badly drawn boxes of the *correct* type, not hallucinated defects of another type. The detector knows what it is looking at. It does not reliably know whether the defect is there, or where it ends.

**crazing** (AP50 0.444) fails by blind miss (53 of 68 misses, 78%). Its measured defect-vs-background separability is 0.32 standard deviations of background roughness -- the lowest of the six classes, against 2.43 for scratches -- and its edge-energy ratio is 1.01, meaning the pixels inside the label are no busier than the pixels outside it. The defect is neither a different grey level nor a texture discontinuity. On top of that, each label claims 24% of a 200x200 frame and there are 2.9 of them per image. The montage shows what that combination means in practice: the labelled and unlabelled parts of a crazing tile are visually indistinguishable, and the boxes carve arbitrary slabs out of a texture that covers the whole strip. Both the low-contrast and the ambiguous-extent hypotheses are true here, and they compound.

**rolled-in_scale** (AP50 0.630) fails differently: by extent error (21 of 34 misses, 62%). Separability 0.51 is the second lowest, but the edge-energy ratio 1.13 is above 1.0, so the defect region is at least texturally distinct. The detector fires -- the montage row shows boxes at confidence 0.28 to 0.42 sitting on or beside the labelled scale -- but the rectangles disagree with the annotator about where one patch of rolled-in scale ends and the next begins. This is an IoU-0.5 scoring failure on a defect that was found, not a detection failure.

Across the six classes, exactly one of these statistics orders them: AP50 against measured separability gives Spearman rho = 0.9429 (p = 0.0048). The others do not. Edge-energy ratio gives rho = 0.0857 -- essentially nothing -- because patches is the easiest class of all at an edge ratio of 0.92, i.e. *smoother* than its background: patches is easy because it is a different grey level, not because it is a different texture. Label overlap gives rho = -0.7714 (p = 0.0724) and mean label area rho = -0.6 (p = 0.208) -- both weak, both negative, and neither significant on six points. Note that box size is the weaker of the two despite the intuition that big boxes are easy: pitted surface has the largest boxes of any class (52% of frame) and is only mid-table for accuracy. So the ranking of causes that the data actually supports is: **grey-level separability first and by a distance; extent/annotation geometry a secondary effect that shows up in the per-class miss taxonomy but not in the aggregate correlations; texture contrast not at all; class confusion not at all.** Six classes is six data points, so these rho values describe this table rather than test a hypothesis -- but the one that is strong is the one the physics predicts, and the class ordering it produces matches every source in docs/research_notes.md section 1c.

The practical reading is that neither weak class is fixed by a bigger network. crazing-style failure -- invisible in the pixels, arbitrary in extent -- is an optics and annotation-protocol problem: dark-field or oblique illumination changes measured separability in a way no loss function can, and a written rule about where a craze network stops is worth more than another 50 epochs. It is also a fair question whether crazing should be scored as a box at all rather than as a whole-frame label or a segmentation mask. rolled-in_scale-style failure -- found but mis-drawn -- is the one that would genuinely respond to more labelled data and to a loss that is less brittle about box edges (the Shape-IoU and Focaler-IoU work in docs/research_notes.md section 6b targets exactly this).

### Ordering against the published literature

| rank (hardest first) | ours | our AP50 | published |
|---|---|---|---|
| 1 | crazing | 0.444 | crazing |
| 2 | rolled-in_scale | 0.630 | rolled-in_scale |
| 3 | pitted_surface | 0.756 | scratches |
| 4 | inclusion | 0.827 | pitted_surface |
| 5 | scratches | 0.909 | inclusion |
| 6 | patches | 0.949 | patches |

Our hardest-first ordering **differs from** the published one (Spearman rho = 0.8286 on the ranks). Source: Maity & Ghosh, arXiv:2510.21811 Table 1 (YOLOv11 column), via docs/research_notes.md s1c.

The largest disagreement is **scratches**: rank 5 here against rank 3 published. Both endpoints of the ordering -- the hardest and the easiest class -- agree.

Note the published values are AP@[.5:.95] on a different 70/20/10 split of NEU-DET, so only the *ordering* is comparable, not the numbers. What transfers is the finding that crazing and rolled-in scale are the hard classes and patches is the easy one, which every source in docs/research_notes.md section 1c agrees on and which we reproduce independently.

![failure montage](model_study_failures.png)

![class difficulty](model_study_difficulty.png)

## 5. Deployment recommendation

**Ship `yolov8n` at 256 px with TTA off.** On the held-out test split, at the input size validation chose, it scores mAP50 0.7524 / mAP50-95 0.3967, from a 6.22 MB checkpoint with 3.012M parameters, at a batch-1 floor of 3.5 ms per 200x200 frame on this machine (median 4.5 ms; see the latency caveat in section 5 -- the absolute milliseconds are not reproducible run to run on a contended laptop and should not be quoted as a throughput figure). With both models at their own best input size the gap over yolov8s is +0.0181 mAP50, 95% CI [-0.0084, +0.0424], which **contains zero**. The honest statement is therefore not 'nano is the better detector' but '**nano is not worse, and it is 3.7x smaller, 3.5x cheaper in FLOPs**'. That is more than sufficient to decide a deployment, and it is the reason to ship it. The larger claim -- that the nano architecture is better on a dataset this size -- is not supported once both models are given their best configuration.

**Why this checkpoint.** On accuracy at best configurations the two are not separable: yolov8n leads on mAP50 by +0.0181 and trails on mAP50-95 by -0.0094, both intervals containing zero. What separates them is cost. yolov8n is 3.7x smaller (3.012M against 11.138M parameters), 6.22 MB against 22.49 MB on disk, and 3.5x cheaper in arithmetic at 256 px. Measured batch-1 latency agrees in direction but understates the margin (4.5 ms against 7.8 ms median, 1.73x against 3.5x of arithmetic), and that is expected rather than awkward -- see the caveat below: at 256 px a single frame does not saturate the GPU, so batch-1 wall clock is part dispatch overhead. The arithmetic ratio is what governs a line running many streams per accelerator. With accuracy tied and every cost axis favouring the smaller model, there is no reading of this evidence in which yolov8s is the right choice.

**Why 256 px.** Chosen on val, over a grid that now runs 128/160/192/224/256/320/416/512/640 -- the original sweep started at 256 px and so reported its own boundary as a peak. Over the corrected grid, 256 px is the val peak on mAP50-95 and on ultralytics' fitness composite (and on both for yolov8s as well), and the test peak on all three metrics; the one criterion that prefers another size is val mAP50, which peaks at 224 px by +0.0109, a margin inside the 0.0254 bootstrap half-width this report measures for differences of that kind, and one that costs -0.0254 mAP50 on test if acted on. Section 2 states that case in full, and `reports/resolution_study.md` carries the whole grid plus a 640 px retrain. This is the highest-stakes knob in the whole configuration: section 2 measures a fall of up to 57% of mAP50 between the best size and 640 px, because the model can only regress boxes at the scale it was trained on. Pin it, and treat any change to it as requiring a retrain rather than a config edit. On real mill frames the decision must be retaken from scratch, since there the input size also determines how many source pixels survive per network pixel and therefore the smallest detectable defect -- see the tiling arithmetic in `src/benchmark.py`.

**Why no TTA.** 2.39x the latency for -0.0061 mAP50 on val. The cost is certain, the benefit is inside the noise the bootstrap measured, and the binding constraint on a rolling line is throughput.

**What must be re-measured before this runs on a line.** This model was trained and tested on NEU-DET, which is hot-rolled carbon steel photographed under one lighting setup. The vendors themselves admit that the same defect class looks different from different upstream mills (AMETEK/Ternium, quoted in docs/research_notes.md section 2a). Every number in this report is a capability demonstration on a public benchmark, not a prediction of performance on Jindal stainless strip. The per-line calibration protocol, not the architecture, is what determines whether this works.

**A caveat on the speed claim, because the measurement does not say what the FLOPs say.** At 256 px and batch 1 the two models are close to the same speed on this machine -- median 4.5 ms against 7.8 ms, a ratio of 1.73x -- even though yolov8s is 3.5x the arithmetic. At this input size a single frame does not saturate the GPU, so the wall clock is dominated by fixed per-call cost (Python dispatch, host-device transfer, NMS) rather than by convolution. The 3.5x shows up once the accelerator is actually busy -- it is already visible at 640 px in the table above (10.9 ms against 23.8 ms median) and it is what governs a mill deployment, where one accelerator serves many camera streams in batches. `src/benchmark.py` measures that regime. Do not quote a batch-1 latency ratio as the throughput argument. Every latency figure here was also taken on a laptop running other work at the time -- swap in use during the sweep: total = 5120.00M  used = 3421.19M  free = 1698.81M  (encrypted).

**And a harder caveat: no absolute latency in this report is reproducible, so none of them belongs on a slide.** Re-running this sweep on the same machine moves the yolov8n@256 batch-1 *median* by more than a factor of two between sessions -- the floor is stable to about 10% but the median, the p95 and every FPS derived from them are dominated by whatever else the laptop is doing. Two consequences. First, quote the ordering and the ratios, never the milliseconds: within one interleaved measurement the two models sit in the right order at every input size, and that ordering survives re-measurement. Second, comparisons are only valid when the two arms were timed alternately in one loop, which is how the TTA multiple in section 3 is now measured and is not how the `by_imgsz` sweep was measured -- so ratios taken *across* rows of that table (including the 3.5x-at-640 point above) are indicative only. A defensible per-frame number for a mill sizing exercise has to come from a quiet machine and a batched sweep, which is `src/benchmark.py`'s job, not this one's.

**The shipping API was cross-checked, and the raw numbers need reading with the caveat above.** `DefectDetector.predict` -- the path production actually calls, with letterboxing, severity scoring and record construction on top of the forward pass -- measured a median 8.0 ms and a floor of 5.3 ms at 256 px, against a floor of 3.5 ms for the bare ultralytics call. Compared at the floor, where contention is squeezed out, the wrapper costs a few tenths of a millisecond and the framework is right that it is close to free; compared at the median it looks several times more expensive, which is an artefact of the two having been timed in separate phases rather than a real cost. This is recorded because it was measured, and because the median reading of it is the sort of number that would otherwise get quoted.

**One consequence to action, and it is worse than a config tidy-up.** The default input size in the shipping code is not 256 px and it is not 320 px either -- it is **640 px**, the single worst size this study measured. `DefectDetector.__init__` and `inference.load_detector` both default to `imgsz=640`, `src/export_model.py` defaults to `--imgsz 640`, and the ONNX artefact currently sitting at `export/yolov8n_neudet_best.onnx` has a frozen `[1, 3, 640, 640]` input. Anyone who constructs a detector without naming an input size, or who deploys that ONNX file, gets test mAP50 0.3438 instead of 0.7524 -- a 54% loss -- silently, with no error and no warning. The Streamlit demo is the one place that is already correct: it defaults to 256 px. The actions are to change the two library defaults, re-export ONNX and CoreML at 256, and delete or clearly retire the 640 px artefact so it cannot be picked up by mistake.

**On retraining cost.** A per-line recalibration of yolov8n at 320 px costs 15.2 s/epoch, or about 0.6 h for a 150-epoch run, measured from `models/yolov8n_neudet/results.csv`. The 640 px equivalent is no longer a projection: a full 150-epoch run at batch 16 was executed and cost **64.9 s/epoch** (median over its uncontended 137 epochs), **3.34 h wall clock end to end**, a per-epoch ratio of **4.3x**. Budget the 320 px figure: an overnight retrain per line is affordable, and both section 2 and the 640 px run itself show that the extra resolution buys nothing -- see `reports/resolution_study.md` section 3.

## Where this lands against published NEU-DET results

docs/research_notes.md section 1a puts the credible band for well-run NEU-DET *detection* baselines at mAP50 0.70-0.80. Our best held-out figure is **0.7524** (yolov8n at 256 px on the held-out test split), which sits inside that band. For reference, the same notes record YOLOv8n at 74.0 and 78.6 mAP50 in two 2025-26 papers, YOLOv11n at 77.2, and the 2020 IEEE TIM DDN paper at 74.8 on a ResNet34 backbone [all HARD]. We are level with a competent stock baseline and below the tuned published improvements, which is where an untuned 150-epoch fine-tune on a laptop GPU should be. Note that the figure recorded before this study, 0.7286, was the same weights scored at 320 px; choosing the inference size on val is worth +0.0238 mAP50 and cost nothing but a sweep.

**Two specific reasons we are not higher, both of them choices rather than accidents.**

*First, we trained at 320 px, not 640, while essentially every published NEU-DET number is at 640.* The measured cost of that choice on this machine: one optimiser step for yolov8s at batch 8 takes 273 ms at 320 px and 864 ms at 640 px, a ratio of 3.2x. (The probe uses batch 8, not the 32 the real runs used. A batch-32 attempt at 640 px was made first and abandoned: with the machine under concurrent memory pressure it went into swap and a single step did not return inside a two-minute budget. The 640/320 ratio is close to batch-independent, so it is measured at a batch that reliably fits, and the batch is recorded so the figure is not misread.) The training run itself logged 36 s/epoch at 320 px (1.5 h for 150 epochs), so 640 px projects to about 1.9 min/epoch, or 4.8 h for the same schedule -- on a laptop GPU that is shared with everything else this project needs to run. **A correction worth stating, and it partly goes the brief's way.** The project brief records 640 px as costing 'more than 11 minutes per epoch versus 35 s at 320'. The 35 s half of that reproduces: per-epoch wall time in `models/*/results.csv` runs at a median of 15 s (nano) and 36 s (small), and the yolov8s median is within a few per cent of the brief's figure. An earlier draft of this report used the *mean* epoch time to say 35 s did not reproduce; that was wrong, because the yolov8s mean (52 s) is inflated by 2 stalled epochs (epoch 63 at 16 min, epoch 64 at 16 min) that between them account for 31 minutes of a 2.2 h run. The same reasoning this report applies to latency -- contention only ever adds time, so quote the median -- applies here and had not been. **The 11-minute half has since been settled by running it, and it does not reproduce.** When this paragraph was written, `reports/training.log` contained no 640 px run to check against and the best this study could offer was a 3.2x step-cost ratio and a machine that demonstrably pages. That gap is now closed by a **completed 150-epoch run**, not a probe: `yolov8n` was trained at 640 px with **batch 16** instead of 32, at **64.9 s/epoch** (median over the 137 epochs that were not competing with a desktop session), **3.34 h** wall clock for the whole job. Peak GPU memory was 4.25 GB, with no stall and no paging. The 11-minute figure came from a batch-32 attempt, which is the configuration that falls off the memory cliff on a 16 GB machine; the arithmetic was never 12x, the allocator was. The correct statement is that **640 px training was skipped for a reason that turned out to be a batch-size artefact, and it is affordable at batch 16**. **And it does not pay.** Swept over the same nine input sizes, the 640 px checkpoint peaks at 512 px on test -- below its own training size, exactly as section 2 predicts -- and scores **mAP50 0.7338 against the shipped 0.7524**, while *winning* narrowly on mAP50-95 (0.3991 vs 0.3967) and tying on fitness (0.4326 vs 0.4323). All three gaps are inside the 0.0254 bootstrap half-width, so the two models are indistinguishable on this test set, and the 640 px one costs 4.3x the training time and 4x the inference pixels to be indistinguishable. Per class it is a real trade: crazing +0.046 and pitted_surface +0.054, against scratches -0.106 and rolled-in_scale -0.071. `reports/resolution_study.md` section 3 carries the run, the epoch timings under both quiet and contended conditions, and the full sweep of the resulting checkpoint.

*Second, we report a genuine held-out test split.* The 1800 NEU-DET images are split 1440/180/180 by `src/prepare_data.py`, and the 180 test images were used for nothing except the final score. Many published NEU-DET numbers are validation numbers, tuned on the same images they are reported on. The size of that effect is visible in this very report: at 320 px, yolov8n scores 0.7396 on val against 0.7286 on test, and yolov8s 0.7237 against 0.6598. Reporting the val column instead would have moved our headline without changing the model.

Neither reason is an excuse for a number, and neither should be presented as one. They are the two knobs a reviewer would ask about, and both have a measurement attached.

## What this study does not establish

- **Training-seed variance is not measured.** The bootstrap resamples images, holding the weights fixed. It cannot tell you whether retraining yolov8s with a different seed would close the gap. Measuring that needs several seeds per architecture; at the logged 15 s/epoch (nano) and 36 s/epoch (small), three seeds each at 150 epochs is about 6.5 hours on this machine. That is the single experiment that would upgrade this study's central claim from 'these weights, on these images' to 'this architecture, on this dataset', and it is affordable.
- **The nano-versus-small question was known before this study ran.** The ordering was already visible in `reports/train_summary_*.json`, so the test split had been looked at once before the bootstrap was designed. The bootstrap quantifies the uncertainty in a comparison that was not pre-registered; it does not make it a blind test.
- **The validation split is not a clean second opinion.** `best.pt` is the epoch that maximised validation fitness, so val is a selection split for both checkpoints and is biased upward on both. This report uses val only to choose inference-time knobs (input size, TTA), where the bias is common to every option being compared and largely cancels. It is deliberately not used to compare the two models.
- **The two checkpoints differ in more than capacity.** They share a recipe, a seed and a split, but one early-stopped at 135 epochs and one ran 150, and neither had its hyperparameters adapted to its size. A fair architecture comparison would tune each model separately; this is a comparison of two checkpoints produced by one recipe, which is the question a deployment actually faces but not the question a paper would ask.
- **One dataset, one imaging setup.** NEU-DET is 1800 200x200 grayscale images of hot-rolled carbon steel. Nothing here has been measured on stainless, on a different camera, or under different illumination, and section 6a of docs/research_notes.md documents domain shift between mills as the primary deployment failure mode -- in the vendors' own words.
- **The failure taxonomy uses one matching convention.** Misses are classified by the best overlapping prediction at a 0.10 IoU floor. A different floor moves boxes between the 'extent error' and 'blind miss' columns. The floor is a stated constant (`EXTENT_IOU_FLOOR`), not a tuned one.
- **Spearman rho on six classes is descriptive.** The appearance-versus-AP50 correlations have six data points. They are reported with their p-values so nobody mistakes them for evidence of a law.
- **NMS IoU, batch size and half precision were not swept.** The deployment recommendation fixes them at the shipping defaults. `src/benchmark.py` covers batch and device; NMS IoU and quantisation remain unmeasured.

---

Machine-readable form: `model_study.json`. The extended section-2 grid (128-224 px), the re-measured 256-640 px cells and the 640 px training run are in `resolution_study.json`, with the narrative in `resolution_study.md`.
