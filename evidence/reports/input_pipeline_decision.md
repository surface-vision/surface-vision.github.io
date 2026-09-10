# Input pipeline: the decision

Closing question for the input pipeline of the Jindal Stainless surface-defect
detector: **what input path gives the best accuracy without damaging the source
image through resampling?**

Two studies ran before this one. `reports/input_study.md` built a genuinely
lossless input path and measured it. `reports/model_study.md` and
`reports/resolution_study.md` swept the input size. This document adds the one
experiment neither of them had, states the decision, and records what was changed
and what was deliberately left alone.

**Decision: nothing about the shipped configuration changes.
`inference.DEFAULT_IMGSZ` stays at 256 px on the framework's ordinary scaled
letterbox. No padded input path is adopted. No model was trained.** The premise
behind the question -- that resampling damages the source and a non-resampling
path should therefore be more accurate -- was tested directly and is false for
this detector. What buys accuracy here is presenting a defect at the
magnification, and through the resampling kernel, that the network was trained
on. Preserving the original pixels buys nothing at all.

---

## 1. What was already known

`reports/input_study.md` established two things that this document builds on.

**The framework was already padding.** The ultralytics validator runs `rect=True,
pad=0.5`, and `BaseDataset.set_rectangle` computes `ceil(imgsz/32 + 0.5) * 32`.
For a square image that is `imgsz + 32`. "Validate at 256" feeds a **288x288**
tensor: a 256x256 resized image inside a 16 px grey (value 114) border. So
`scale N` and `pad N` are a matched pair -- same tensor, same compute, same border
colour, and the only difference is whether the 200x200 source was interpolated up
to N first.

**The lossless path works, and loses.** A 200x200 NEU-DET frame dropped verbatim
into a 224 or 256 canvas is byte-exact (verified on 180/180 images, box round-trip
error 1e-08 px) and costs 0.0238 to 0.0521 mAP50 against the scaled path at the
same tensor size, on 8 of 8 (checkpoint, split, canvas) configurations.

That study stated its own confound clearly: both checkpoints were fine-tuned on
scaled inputs, so it measured an *inference-time substitution*, not the value of
padding in principle. That confound is what this document had to resolve before a
decision could be made honestly.

## 2. The experiment that was missing

The pad-versus-scale contrast moves two things at once:

* **fidelity** -- pad keeps every source pixel; bilinear scaling smooths them and
  invents intermediate grey levels;
* **magnification** -- pad shows the defect at 1.00x against the network's fixed
  stride grid; scaling to 256 shows it at 1.28x.

The project premise is a claim about *fidelity*. So separate them, by adding an
arm that is lossless **and** magnifying:

> **Nearest-neighbour magnification.** `cv2.resize(src, (256, 256),
> INTER_NEAREST)` on a 200x200 source is exactly invertible. The scale factor is
> greater than 1, so every one of the 200 source indices is hit by at least one of
> the 256 output indices, and the source is recovered **byte for byte** by pure
> indexing -- asserted in code, on a real frame and again on a synthetic one in
> `tests/test_input_pipeline.py`, not argued. It invents nothing and discards
> nothing. It is every bit as lossless as padding, and it magnifies by 1.28x
> exactly like the shipped path.

Five arms, all at one fixed 288x288 network tensor, all on the same 180 images:

| arm | what reaches the network | lossless? | magnification |
|---|---|---|---|
| `scale` | the framework's own in-loader bilinear upscale | no | 1.28x |
| `bilinear` | the same upscale, precomputed on disk as PNG | no | 1.28x |
| `nearest` | pixel-replicated upscale, source exactly recoverable | **yes** | 1.28x |
| `decimated` | 200 -> 160 (INTER_AREA) -> 256, 36% of samples destroyed | no, badly | 1.28x |
| `pad` | the 200x200 source verbatim in a grey canvas | **yes** | 1.00x |

Two controls make the arms readable:

* **`bilinear` is an identity check on the instrument.** It runs the identical
  `cv2.resize` call `BaseDataset.load_image` makes for an upscale, then hands the
  loader a 256x256 file whose own scale factor is 1.0, so the loader does nothing.
  If pre-resampling on disk is neutral, it must reproduce the framework's `scale`
  number to the last digit. **It does, on every checkpoint and split** (see the
  table below: 0.7524/0.7524 and 0.7642/0.7642 and 0.7466/0.7466, and mAP50-95,
  precision and recall agree to 1e-9). Everything else in the probe is therefore
  measuring the kernel, not the substitution.
* **`decimated` is a sensitivity check on the benchmark.** It genuinely destroys
  information -- a 0.8x INTER_AREA decimation removes 36% of the source samples
  and no operation can bring them back -- at the *same* magnification as the other
  two. It exists so that a null result on `nearest` could not be dismissed as
  "180 images cannot see a fidelity difference at all".

Also verified before any accuracy number was read: all 1080 written PNGs survive
the write-and-read-back round trip byte-exact; and the network tensor really is
the file on disk, unscaled, in a 114 border, for 180/180 images on every derived
split (pulled out of the dataset object the validator itself builds).

## 3. Result: losslessness does not predict accuracy

Held-out **test** split, 180 images, 446 instances, `imgsz` 256 (288 px tensor),
ultralytics validator defaults (conf 0.001, NMS IoU 0.7, max_det 300), cpu.

| arm | yolov8n_neudet mAP50 | mAP50-95 | yolov8n_joint mAP50 | mAP50-95 |
|---|---|---|---|---|
| `scale` (shipped) | **0.7524** | 0.3967 | **0.7642** | 0.4008 |
| `bilinear` (identity check) | 0.7524 | 0.3967 | 0.7642 | 0.4008 |
| `pad` (lossless, 1.00x) | 0.7091 | 0.3874 | 0.7164 | 0.3695 |
| `decimated` (36% destroyed) | 0.6910 | 0.3286 | 0.7296 | 0.3749 |
| `nearest` (lossless, 1.28x) | **0.6252** | 0.3036 | **0.6257** | 0.3027 |

**val** split (the split a choice may legitimately be made on), `yolov8n_neudet`:
`scale` 0.7466, `bilinear` 0.7466, `pad` 0.7390, `decimated` 0.7139, `nearest`
0.6464.

Read the ordering, not the gaps:

* The **best** arm is a lossy one, on every checkpoint and every split.
* The **worst** arm is a lossless one, on every checkpoint and every split -- and
  it is not the padded one. Nearest-neighbour magnification preserves the source
  perfectly and costs 0.100 to 0.138 mAP50 against bilinear magnification of the
  identical pixels.
* An arm that **destroys 36% of the source samples** beats the perfectly lossless
  `nearest` arm by 0.066 to 0.104 mAP50, and on the joint checkpoint it beats the
  perfectly lossless `pad` arm too (0.7296 against 0.7164).

There is no relationship between how much of the source survives and how well the
detector does. A CNN does not consume bits; it consumes a texture distribution at
a scale. Nearest-neighbour replication keeps every bit and produces blocky,
staircase-edged high-frequency content that nothing in training looked like, and
the network's early filters -- which on this dataset are exactly the filters that
separate crazing from pitting -- respond to it wrongly. Smooth decimation loses a
third of the samples and still produces an image that *looks like* the training
distribution.

**This retires the question the project set out to answer.** "Best accuracy
without damaging the source through resampling" presupposes that damage is the
mechanism. It is not. Of the five arms measured, the two byte-exact ones finish
3rd or 4th, and last.

### The padded path, re-read

With fidelity eliminated as an explanation, what is left for the padded path is
its magnification. The check: compare it against the scaled resolution grid
evaluated at the *same* content magnification. Interpolating the published
`scale192`/`scale224` cells of `reports/input_study.json` to a 200 px content size:

| checkpoint | split | scaled curve at 200 px content | measured `pad256` | difference |
|---|---|---|---|---|
| yolov8n_neudet | test | 0.7231 | 0.7091 | -0.0140 |
| yolov8n_neudet | val | 0.7422 | 0.7390 | -0.0031 |
| yolov8n_joint | test | 0.7358 | 0.7164 | -0.0194 |
| yolov8n_joint | val | 0.7739 | 0.7497 | -0.0242 |

The padded path scores, within 0.025 mAP50 everywhere, exactly like a 200 px
input -- which is what it is. It is not being punished for its grey border; it is
simply showing the network a smaller defect. (This is an interpolation of two
measured cells, not a measurement, and on the flat val curve the residual is
inside the noise either way. It is offered as the size of the remaining effect,
not as a result.)

## 4. The decision, candidate by candidate

**Change `DEFAULT_IMGSZ`: no.** Every size from 224 to 320 px is within 0.0254
mAP50 of its own curve's best on all four (checkpoint, split) curves, and the
paired bootstrap (1000 resamples of 180 images, seed 1337) puts zero inside every
interval in that band. The val peak at 224 px that an audit flagged is a maximum
selected from 8 sizes; corrected for that selection it is 99.3% [-0.0057,
+0.0479] and contains zero. There is no evidence to move off the incumbent, and
256 sits in the middle of the plateau rather than at its edge, which is the
safest place to stand when the next checkpoint shifts the curve slightly. What
matters in this knob is staying off the ends -- the grid ends give up as much as
0.1075 mAP50, and inheriting the framework's 640 costs 0.41.

**Adopt a pad-instead-of-scale input path: no.** It loses on 8 of 8
configurations in `reports/input_study.md` and on both checkpoints again here.
Its one advantage over the incumbent -- byte-exactness -- has now been measured
and is worth nothing: the other byte-exact path in this study is the worst arm of
five.

**Train a model with a padded pipeline so training and inference match: no, and
the training budget was deliberately not spent.** See section 5.

**Make `models/yolov8n_joint` the default: no, and not because of this study.**
That decision is argued in full in the `resolve_weights` docstring in
`src/inference.py`, and the analysis there is sound and is not changed by
anything measured here: the joint model is a tie in domain (+0.0118 mAP50 here,
inside a bootstrap interval that contains zero) and decisively better out of
domain, but three consumers still assume a six-class head
(`demo/app.py:preferred_checkpoint_index`, `src/false_alarm.py` ~line 892,
`src/export_model.py` line 389) and flipping the default alone would regress them
silently. Note for the record that this study scored the joint checkpoint on
every arm: it puts the shipped scaled path first, both byte-exact paths below it,
and `nearest` last, exactly as the shipped model does (the two middle arms swap
places). So the input-pipeline decision does not depend on which checkpoint is
eventually promoted.

**Conclude the current configuration is right and change nothing: yes.** With the
qualification that "nothing" means no behaviour changed. The evidence, the
reasoning and the tests that pin it down did change, and that is the deliverable.

## 5. Why no model was trained

The brief offered 60-75 minutes of wall clock to train a model on the padded
pipeline, conditional on the evidence supporting a padded, non-resampling input
path. It does not support it, so the run was skipped. Specifically:

1. **The reason to prefer padding has been falsified.** Padding was worth trying
   because it is lossless. Losslessness has now been measured against a second
   lossless path and against a deliberately lossy one, and it does not predict
   accuracy in either direction. Training a model to be good at a pipeline whose
   only claimed advantage does not exist is not a promising use of an hour.
2. **What the padded path would have to overcome is not the border, it is the
   magnification.** The padded arm already scores like a 200 px input (section 3).
   The best case for a pad-trained model is that it recovers the entire residual
   in that table and lands on the scaled curve at 1.00x content -- roughly 0.723
   (neudet) and 0.736 (joint) on test, still below the 0.7524 and 0.7642 the
   shipped path gets at 1.28x. Retraining changes the model's priors; it does not
   put more defect pixels on the stride grid.
3. **"Training and inference must match" is already known to be false in this
   repository.** Both shipped checkpoints were fine-tuned at `imgsz: 320`
   (`models/*/args.yaml`) and both are more accurate at 256 than at 320 -- test
   mAP50 0.7524 against 0.7286 for `yolov8n_neudet`. The best inference
   magnification is a property of the defect size against the network's stride
   grid, not of the size the weights were trained at, so "the model never saw a
   padded border" is a weaker explanation of the pad deficit than it first looks.
   Training augmentation covers part of it too: `mosaic: 1.0, scale: 0.4,
   translate: 0.1, degrees: 10.0`, with ultralytics filling mosaic canvases and
   warp borders with the same value 114.
4. **Cost on the other side of the ledger.** Adopting a padded pipeline would
   move the runtime off the framework's standard path and invalidate the ONNX and
   CoreML exports (`reports/export_summary.json`), the fitted operating point
   (`reports/operating_point.json`, conf 0.15) and the calibration
   (`reports/calibration.json`), all of which were fitted on the scaled path.
   That is a large bill for a configuration whose best case is a tie.

**What would legitimately reopen this.** Fine-tune from
`models/yolov8n_neudet/weights/best.pt` on `data/input_study/pad224` (already
built, byte-exact, 6- and 10-class yamls present) and validate the padded path
against `scale256`. If a pad-trained model exceeds 0.7524 test mAP50 on
`yolov8n_neudet` with an interval that excludes zero against the incumbent, this
decision is wrong and should be reversed. Nothing measured here forbids that
result; it is simply not the way the evidence points, and the brief's condition
for spending the compute was not met.

Two smaller open items, stated so they are not mistaken for settled:

* Everything here is validation geometry (a 288 px tensor with a 16 px border).
  The deployed `predict()` path letterboxes a single 200x200 frame to a 256 px
  tensor with no border at all. The accuracy *ordering* is expected to carry over
  and the same bilinear upscale is used, but the deployed path has not been swept
  size by size.
* 180 images is a small benchmark. It resolves the differences in this document
  comfortably -- 0.13 mAP50 against interval half-widths near 0.03 -- but it
  cannot resolve 224 from 320, which is exactly why no size change was made.

## 6. Interval table

Paired percentile intervals on mAP50 differences: 1000 resamples of the 180
images, seed 1337, every arm scored on the same resampled image sets inside one
loop, so the between-image variance they all share cancels in the difference.
Same design, same seed and same resample count as `reports/input_study.md`.

| contrast | yolov8n_neudet / test | yolov8n_neudet / val | yolov8n_joint / test |
|---|---|---|---|
| `bilinear - scale` (instrument identity) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| `nearest - bilinear` (fidelity, magnification fixed) | **-0.1273** [-0.1584, -0.0915] | **-0.1003** [-0.1250, -0.0736] | **-0.1384** [-0.1752, -0.0982] |
| `nearest - pad` (both lossless) | -0.0839 [-0.1164, -0.0464] | -0.0927 [-0.1149, -0.0647] | -0.0907 [-0.1257, -0.0556] |
| `decimated - bilinear` (36% of samples destroyed) | -0.0614 [-0.0867, -0.0313] | -0.0327 [-0.0646, +0.0014] | -0.0345 [-0.0560, -0.0083] |
| `pad - scale` (`input_study`'s headline) | -0.0434 [-0.0688, -0.0195] | -0.0076 [-0.0296, +0.0172] | -0.0477 [-0.0733, -0.0175] |

Three things to read off it.

1. **The instrument is neutral.** The identity contrast is exactly zero with a
   zero-width interval, on all three (checkpoint, split) groups.
2. **The headline is resolved everywhere and is enormous by this benchmark's
   standards.** `nearest - bilinear` is -0.100 to -0.138 mAP50, with intervals about
   0.06 wide, against a benchmark that cannot resolve 224 px from 320 px. A
   lossless transform of the same pixels costs between 2 and 18 times what the
   entire pad-versus-scale question is worth (`pad - scale` runs -0.0076 to -0.0477).
3. **This study reproduces the previous one exactly.** The `pad - scale` row was
   measured independently here and agrees with `reports/input_study.json` to four
   decimal places on point estimate and on both interval ends, in all three
   groups. Same instrument, same answer.

Ranking of the five arms. `yolov8n_neudet` on test: `scale` = `bilinear`
(0.7524) > `pad` (0.7091) > `decimated` (0.6910) > `nearest` (0.6252).
`yolov8n_joint` on test: `scale` = `bilinear` (0.7642) > `decimated` (0.7296) >
`pad` (0.7164) > `nearest` (0.6257). The lossy shipped path is first on both;
the two byte-exact paths finish 3rd or 4th, and last.

## 7. What changed, and what did not

**Changed**

| file | change |
|---|---|
| `src/inference.py` | The comment above `DEFAULT_IMGSZ` now cites `reports/model_study.json`, `reports/input_study.md`, `reports/input_pipeline_probe.json` and this document, and records the nearest-neighbour result so the lossless idea is not rediscovered and re-implemented. **No code and no value changed.** |
| `src/input_pipeline_probe.py` | New. The experiment in section 2, sharing `src/input_study.py`'s validator, per-image statistics capture, bootstrap and seed. |
| `tests/test_input_pipeline.py` | New. 29 tests pinning the constant, the instrument identity, the invertibility of the nearest-neighbour arm (re-proved on a synthetic frame, not quoted), the ordering that carries the decision, and the numbers this document quotes. |
| `reports/input_pipeline_probe.json` | New. Raw measurements. |
| `reports/input_pipeline_decision.md` | This document. |
| `data/input_study/{nearest,bilinear,decimated}256/` | New derived splits, 114 MB of PNG, val and test. Measurement inputs only -- nothing in `src/` reads them, and a test asserts `src/inference.py` contains neither `data/input_study` nor any derived split name. |

**Not changed, deliberately**

* `inference.DEFAULT_IMGSZ` -- stays 256.
* `inference.resolve_weights` -- still serves `models/yolov8n_neudet`. The
  recommendation to promote the joint checkpoint, and the four items that must
  land first, remain as written in its docstring by the previous task.
* `DefectDetector.predict_tiled` -- the tile default stays `imgsz`, which is the
  only setting that keeps one source pixel on one network pixel. This study
  reinforces it rather than changing it.
* `demo/app.py` -- not owned by this task, and not touched.
* No model trained, no export regenerated, no operating point or calibration
  re-fitted -- none of their inputs moved.
* The 640 px training job and the Streamlit demo on port 8501 were left running
  throughout; the probe ran on cpu specifically to stay off the GPU.

## 8. Verification

**The shipped model still reproduces its published held-out number.** Run
independently of the probe, straight through the framework, against
`data/neu-det/data.yaml`:

```
$ .venv/bin/python -c "
from ultralytics import YOLO
m = YOLO('models/yolov8n_neudet/weights/best.pt')
r = m.val(data='data/neu-det/data.yaml', split='test', imgsz=256, device='cpu',
          plots=False, verbose=False)
print('mAP50', round(float(r.box.map50), 5)); ..."

SHIPPED CHECKPOINT, held-out NEU-DET test, imgsz 256, cpu
mAP50     0.75245
mAP50-95  0.39671
precision 0.696
recall    0.68671
```

`reports/evaluation.json`, generated on mps before this task began, records
mAP50 0.7524475543783158, mAP50-95 0.3967094751925778, precision 0.696003746864622,
recall 0.6867120962219055. Identical, across a device change.

**The suite passes**, 520 tests against the 491 that passed before this task (the
29 new ones are `tests/test_input_pipeline.py`):

```
$ .venv/bin/python -m pytest tests/ -q
520 passed, 1 skipped, 2 warnings in 54.70s
```

The one skip is pre-existing. One test failed on the first attempt --
`test_no_derived_study_dataset_leaked_into_the_runtime`, a test of this task's own
writing that forbade the string `input_study` anywhere in `src/inference.py` and
so was tripped by the new comment *citing* `reports/input_study.md`. It was
narrowed to forbid the dataset directory and the derived split names, which is
what it was for, and it now passes.

The 640 px training job (`models/yolov8n_640`) and the Streamlit console on port
8501 were running throughout and were not disturbed; the console still answered
HTTP 200 after the probe. Every measurement in this document ran on cpu.

---

## 9. Plain English, for a non-specialist

### Why blowing a 200 pixel image up to 640 destroys accuracy

The photographs in this dataset are 200x200 pixels. YOLO insists its input be a
multiple of 32, and its default is 640. So the obvious thing -- feed it 640 --
means enlarging every photo 3.2x before the network sees it.

Enlarging adds no information. There is no extra detail hiding in a 200 pixel
photo; the enlargement just makes the same detail bigger and blurrier, the way a
300% photocopy does. But it does change something the network cares about a great
deal. A detector does not look at a picture as a whole; it looks at it through a
fixed grid, making decisions on patches of 8, 16 and 32 pixels. What matters is
how big a defect is **in grid squares**. Enlarge the photo and a scratch that used
to sit neatly in the fine part of the grid spills across the coarse part, which
was built for large blotches, not hairlines. On top of that, the model learned
what steel texture looks like at one scale; at 3.2x it is being shown a world it
has never seen.

The cost is not subtle. Same model, same photographs, only the input size
changed:

| input size | 160 | 224 | **256** | 320 | 416 | 512 | 640 |
|---|---|---|---|---|---|---|---|
| test mAP50 | 0.659 | 0.727 | **0.752** | 0.729 | 0.623 | 0.484 | 0.344 |

(`yolov8n_neudet` on the 180 held-out test images. 160-320 px from
`reports/input_study.json`, 416-640 px from `reports/model_study.json`; the
two studies agree to 0.00e+00 mAP50 on the ten cells they share.)

Accepting the framework's default would have thrown away more than half the
system's accuracy, with no error message and nothing in the logs to show for it.

### Why we tile a 2048 pixel line-scan camera instead of shrinking the frame

A real inspection line does not produce 200 pixel snapshots. A line-scan camera
produces frames thousands of pixels wide -- 2048 in the strips this project tests
against.

The lazy option is to shrink the whole frame down to one network input. Measured
on a real 2048x1000 frame, that is a single resize to 256x125: **1.56% of the
frame's pixels reach the network**. A hairline scratch three pixels wide becomes
a third of a pixel. It is gone before the detector runs.

So the system does the opposite. It cuts the frame into 256x256 tiles at native
scale, with 20% overlap, runs each one, and then merges the results with a single
pass of non-maximum suppression across the whole frame so a defect crossing a
tile seam is counted once rather than twice. At the default tile size this path
touches nothing: verified on a real frame, **zero resize calls in a 50-tile pass,
and every tile byte-identical to the region of the photograph it came from**.

The difference on labelled strips is the whole argument: shrinking the frame
localises **1 of 6** labelled defects; tiling localises **5 of 6**.

This also explains why the two answers are not in conflict. Shrinking a big
picture throws real detail away, and that is fatal. Enlarging a small picture
throws nothing away, and it turns out to help, up to a point. Different
operations, different consequences.

### So what is the ideal input size for this data?

**256 pixels -- a 1.28x enlargement of the 200 pixel source -- and the honest
version of that answer is that anything from 224 to 320 is equally good.**

With 180 test images the measurement is not sharp enough to separate the sizes
inside that band; it is sharp enough to separate the band from everything outside
it. 256 is chosen because it sits in the middle of the plateau rather than on its
edge, so a future checkpoint that shifts the curve slightly does not fall off.

The genuinely surprising part is what we tried and rejected. Since 200x200 fits
inside a 224x224 box with a 12 pixel border, you can feed the network the
original photograph with **no enlargement at all**, every pixel exactly as the
camera recorded it. We built that, proved it byte-exact, and it was *worse* -- by
about 0.03 to 0.05 mAP50.

To find out why, we built a second path that keeps every original pixel *and*
enlarges: nearest-neighbour, which simply repeats pixels rather than blending
them. The original photograph can be recovered from it exactly. It scored
**0.625** -- the worst of everything we tried. Meanwhile a path that deliberately
threw away a third of the original pixels, and smoothly restored the size, scored
**0.691 to 0.730**.

The conclusion is short, and it is the one to take to a judge: **the detector does
not care how many original pixels survive. It cares that the image arrives at the
size, and with the smoothness, that it was trained on.** Perfect preservation is
an appealing idea that this data does not reward. The right question was never
"how do we avoid touching the pixels" -- it was "how do we present a defect at the
scale the network reads best", and for a 200 pixel frame the answer is a modest
1.28x enlargement, while for a 2048 pixel line-scan frame the answer is to stop
resizing altogether and tile.

---

## 10. Reproducing

```
.venv/bin/python src/input_pipeline_probe.py --resamples 1000
.venv/bin/python -m pytest tests/ -q
```

Raw numbers: `reports/input_pipeline_probe.json`. Prior work this rests on:
`reports/input_study.md` (pad versus scale, the resolution grid, the bootstrap),
`reports/model_study.md` and `reports/resolution_study.md` (the size sweep and
the 640 px collapse), and the lossless-tiling contract in the `predict_tiled`
docstring in `src/inference.py`.
