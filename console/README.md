---
title: Surface Inspection Console
emoji: 🔶
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Steel strip surface defect detection, with the caveats attached
tags:
  - computer-vision
  - object-detection
  - yolov8
  - manufacturing
  - quality-inspection
  - streamlit
---

# Surface Inspection Console

An AI surface-defect inspection station for stainless steel strip, running the real
inference stack rather than a picture of one. Drop in a frame and it gates the
frame, scores it, localises every defect, prints a calibrated probability, and
hands back the corrective action a mill operator would take -- or refuses to answer
and says why.

The model is a 6.22 MB YOLOv8n fine-tuned on NEU-DET. **This is a capability
demonstration on public research datasets, not a Jindal Stainless production
system**; no line data was used and nothing here has seen a calibration target.

---

## What it does

**Single frame** -- one image in, a PASS / DEFECT / VERDICT-WITHHELD verdict out,
the source next to the annotated overlay, a per-detection table in pixels and
millimetres, and the mill root cause and standing corrective action for every
defect family present. Four things happen around that verdict:

1. **A plausibility gate runs before the detector.** A six-class defect head has
   no "not steel" answer, so a logo or a screenshot otherwise comes back as a
   confident critical disposition -- measured: an all-white frame scores
   `pitted_surface 0.739` at severity 86.3 critical on the shipped model. Six
   physical checks (resolution, aspect, focus, tonal depth, saturation, clipping)
   decide whether the frame is strip imagery at all. A frame it refuses gets a
   **withheld verdict with the failing checks on screen**, not a silent
   suppression, and an explicit override so the failure mode can be shown.
2. **Tiling engages on its own** when one network input cannot represent the
   frame -- wide aspect, or a large downscale -- with a visible badge naming the
   trigger. On the 2048x1000 line-scan frame shipped here that is the difference
   between **0 detections whole-frame and 20 tiled**.
3. **The confidence is calibrated.** The number an operator acts on is
   P(this box is a true positive) from an isotonic map fitted on the validation
   split, with the raw detector score kept beside it and both labelled.
4. **An Explain button** renders an EigenCAM attention map on demand.

**Batch inspection** -- many frames through the batched path, a sortable frame
table, the defect-family distribution, CSV exports for the coil file, and the
**per-coil disposition report**: one self-contained HTML file with the
ACCEPT / DOWNGRADE / HOLD decision, the rule that produced it, the severity
distribution, the coil position map and the worst frames with their overlays.

**Line simulation** -- frames scored one at a time, the way a line camera feeds
them, for real single-shot latency and a running defect rate. The
field-of-view advance rate is deliberately *not* called a line speed, and the
accelerator count a full-width line would actually need is printed next to it,
read out of the capacity model in `reports/benchmark.json`.

**Defect atlas** -- the standing knowledge base, useful before anything is
uploaded.

---

## How to use it

1. The console opens on a held-out test frame with a verdict already on screen.
   Nothing needs to be uploaded.
2. **Defect family** picker -- six families, ordered by measured median peak
   confidence rather than alphabetically, each with its own figure printed
   underneath. `crazing` is last on purpose: median peak confidence 0.29, and 93%
   of its frames score under 0.50. That is a hard class, stated, not hidden.
3. **Frame source -> Wide strip capture** -- a real 2048x1000 line-scan frame.
   Watch the badge change to `TILED` and the detection count go from 0 to 20.
4. **Frame source -> Upload** -- any JPG, PNG, BMP or TIFF. Greyscale, palette,
   CMYK and 16-bit sources are converted; EXIF orientation is honoured. Try
   something that is *not* steel and watch the verdict be withheld.
5. **Batch inspection -> Load sample batch -> Build report** -- the coil file.
6. **Sidebar** -- confidence and NMS thresholds, network input size, tiling
   policy, the optical-scale assumption behind every millimetre on screen, and a
   second checkpoint to compare against.

---

## Measured numbers

Every figure below is traceable to a file in the project repository, named beside
it. Nothing on this page is an estimate.

### The shipped detector -- `models/yolov8n_neudet/weights/best.pt`

YOLOv8n, 3.012 M parameters, 1.312 GFLOPs at 256 px, 6.22 MB. Fine-tuned at
320 px, deployed at 256 px. Held-out NEU-DET test split, 180 frames never
trained on and never tuned on:

| | measured | source |
|---|---|---|
| mAP50 | **0.7524** | `reports/model_study.json` -> `runs[]` |
| mAP50-95 | **0.3967** | same row |
| precision / recall | **0.696 / 0.687** | same row |
| defect-type accuracy, dominant detection | **98.9%** (178/180) at conf 0.05; **97.2%** at conf 0.15 | `reports/evaluation.json` -> `image_level`; `reports/evaluation.md` section 3 |
| defective frames flagged at conf 0.15 | **92.2%** (166/180) | `reports/false_alarm.json` -> `holdout_detection` |
| boxes raised on steel crops carrying no annotated defect | **<= 23.7%**, 95% CI [17.7%, 30.1%] | `reports/false_alarm.json` -> `clean_patch_false_alarm` |
| class confusions among missed labels | **0** of 168 | `reports/model_study.json` -> `failures.taxonomy_rows` |

Per-class AP50: patches 0.949, scratches 0.909, inclusion 0.827,
pitted_surface 0.756, rolled-in_scale 0.630, crazing 0.444
(`reports/model_study.json` -> `per_class_AP50`). The spread is not noise: AP50
against a measured foreground/background separability score has a rank
correlation of **0.943** (p = 0.0048), so the model is weak exactly where the
defect is hardest to see.

**Operating point** conf **0.15**, NMS IoU **0.45**, 256 px -- chosen on the
*validation* split by a mill cost model priced at 12 unnecessary re-inspections
per escaped defect and 5% defective-frame prevalence, and binding on a 90%
detection floor rather than on the economics
(`reports/operating_point.json`).

### Calibration

Isotonic regression, 34 knots, fitted on 180 validation frames. Expected
calibration error on the held-out test split **0.1418 -> 0.0461**, a 67.5%
reduction. The map is strictly increasing, so it reorders nothing: mAP50 is
numerically unchanged at 0.707986 before and after
(`reports/calibration.json`, `reports/calibration.md`).

### The out-of-distribution gate

Six physical checks, thresholds each set from a measurement:
**0 false rejections over 3,200 genuine steel frames** (1,800 NEU-DET +
1,400 Severstal) and **13 of 15** correct on a held-out synthetic-negative suite
(`reports/ood_guard.json`, `reports/ood_guard.md`). It tests physics, not
semantics -- a monochrome in-focus texture that is not steel will still get
through, and the console says so.

### Resolution, and why 256 px

Accuracy falls steeply above the 320 px training size: test mAP50 **0.7524** at
256 px, 0.7286 at 320, 0.6230 at 416, **0.3438** at 640 -- a 54% collapse
(`reports/model_study.json`). Retraining at 640 px does not recover it: that run's
own best input reaches 0.7338, which is 0.0186 *below* the shipped model and
inside the +/-0.0254 bootstrap band (`reports/resolution_study.md`). The console
still offers the larger inputs, because a full-width strip capture is genuinely
a different input from a 200x200 crop.

### The second checkpoint -- `models/yolov8n_joint/weights/best.pt`

Also shipped, also selectable, deliberately **not** the default. Trained on
NEU-DET plus Severstal with the NEU-DET class indices unchanged; 10 classes.

| | shipped | joint | source |
|---|---|---|---|
| in-domain mAP50 (NEU-DET test, 256 px) | 0.7524 | **0.7642** | `reports/gap1_detection_metrics.json` |
| clean-frame false alarms, real defect-free strip | 93.7% | **32.5%** | `reports/gap1_cross_domain.json` |
| boxes per clean tile | 2.02 | **0.070** | same |
| defect-vs-clean ROC AUC, cross-domain | 0.608 [0.576, 0.640] | **0.958** [0.948, 0.968] | same |
| cross-domain recall | 0.720 | **0.866** | same |
| clean-coil disposition | HOLD, 55.2% defect rate | **DOWNGRADE, 4.8%** | `reports/cross_domain_clean_coil_*.json` |

It is the better model on real clean steel by a wide margin. It is not the
default because four of its ten classes are Severstal mask values 1-4, for which
that dataset publishes no defect type at all, so they carry a placeholder
severity tier and no mill root cause; because its in-domain crop-level
separation moves slightly the *wrong* way (AUC 0.968 -> 0.940, intervals
overlapping, unexplained); and because the calibration map and the operating
point were both fitted on the shipped checkpoint. Select it and the console says
all of that in the sidebar rather than quietly applying the wrong map. The four
conditions for promoting it are written out in `vendor/inference.py` ->
`resolve_weights`.

### Latency on CPU

Measured by `tools/bench_cpu.py` through the vendored code on the project's
development host, CPU only, torch pinned to 2 threads to approximate this
Space's 2 vCPU. Reproduce on the Space's own hardware with
`python tools/bench_cpu.py --threads 2 --cam`:

| | mean | p50 | p95 | |
|---|---|---|---|---|
| plausibility gate, 200x200 | 0.3 ms | 0.3 | 0.4 | numpy + OpenCV, no torch |
| single frame, 200x200 | **6.5 ms** | 6.5 | 6.6 | 153 fps |
| batched, per frame | 4.2 ms | 4.2 | 4.4 | 236 fps |
| 2048x1000 strip, whole frame | 8.7 ms | 8.7 | 8.9 | **0 detections** |
| 2048x1000 strip, 50 tiles at 256 px | **194 ms** | 195 | 195 | **20 detections**, 3.9 ms/tile |
| EigenCAM, 200x200, first call in a process | 1,585 ms | | | then **11 ms** |
| EigenCAM, 2048x1000 strip, first call | 14.8 s | | | measured through the UI |

Two honest notes on those figures. First, the same benchmark at 8 torch threads
is **slower** than at 2 (7.3 ms vs 6.5 ms single-frame; 244 ms vs 194 ms tiled) --
for a 3 M-parameter model the thread overhead dominates, which is why the app
pins torch to the CPUs the container actually has rather than to what
`os.cpu_count()` reports. Second, a shared Spaces vCPU is slower per core than
that host's, so treat every number above as a floor and read the timings the
console prints on screen instead: those are measured here, in your request.

### Regression

`540 passed, 8 skipped` over the project's test suite, re-run 2026-09-10, with
the shipped model still reproducing mAP50 0.7524 to eight significant figures
after everything above.

---

## Datasets and licences

**NEU-DET** -- Northeastern University, hot-rolled steel strip surface defects.
200x200 greyscale-toned crops, six classes. Upstream ships 1,620 train /
180 test; the project carves a stratified 180-frame validation set out of train
and leaves the 180-frame test split fully held out (1,440 / 180 / 180, verified
free of cross-split duplicates by content hash). *Licence:* the release states
none. It is distributed for academic research and is treated here as
research-use-only; the project records no licence grant for it, and this Space
ships 18 frames of the held-out test split for demonstration, with attribution.
Cite: He, Song, Meng, Yan, *An End-to-End Steel Surface Defect Detection
Approach via Fusing Multiple Hierarchical Features*, IEEE Trans. Instrum. Meas.
69(4):1493, 2020.

**Severstal Steel Defect Detection** -- used only to train the second checkpoint
and to measure false alarms on verified defect-free strip; 1600x256 frames, mask
values 1-4 with **no published defect semantics**, which is why those four
classes are named after their mask value and carry no root cause. Accessed via
the public `Voxel51/severstal_steel_defects` mirror of the Kaggle competition
data. *Licence:* competition terms; no images from it are redistributed in this
Space.

**GC10-DET** -- Lv, X.; Duan, F.; Jiang, J.J.; Fu, X.; Gan, L. *Deep Metallic
Surface Defect Detection: The New Benchmark and Detection Network.* Sensors 2020,
20(6), 1562. Real 2048x1000 greyscale line-scan strip frames. *Licence:*
**CC BY 4.0** -- attribution required and given, here and in `assets/README.md`.
Accessed via the `imaadd05/gc10-det` mirror, which carries the explicit CC BY 4.0
grant (the original Baidu Pan release states no licence). One unmodified frame is
redistributed in this Space under that licence. **The model shipped here has
never been trained on GC10**, so the strip demonstration shows the inference path
on real strip geometry and is not an accuracy claim on that dataset.

---

## Honest limitations

**It has never seen a length of sound steel.** NEU-DET contains no defect-free
image in any split -- all 1,800 frames carry at least one labelled defect. Every
defect rate this console computes over sample frames is therefore near 100% by
construction and says nothing about how often the detector would flag good
material. The false-alarm question was answered elsewhere, on mined clean crops
and on real Severstal strip, and the answer for the *shipped* checkpoint is
uncomfortable: it fires on 93.7% of verified defect-free frames. That is the
single largest gap in this system, it is why the joint checkpoint exists, and it
is not fixed in the default.

**The clean-crop false alarm rate is an upper bound.** 23.7% is measured on
crops carrying no annotated box, which is not the same as verified clean. 78.2%
of the boxes raised on them carry the source image's own defect class against a
16.7% chance rate, so an unknown share are unlabelled continuations of a real
defect rather than hallucinations on sound metal.

**Millimetres are an assumption, not a measurement.** NEU-DET frames carry no
scale bar and nothing in this project has ever seen a calibration target. The
optical scale is an operator-editable number in the sidebar; change it and every
millimetre on screen changes with it, which is the point.

**Severity is an ordinal triage aid.** Class base tier scaled by confidence and
box area. It is not calibrated against mill outcomes, and the disposition
thresholds are conservative constants that a quality department would own.

**Crazing is weak.** AP50 0.444; median peak confidence 0.29; 93% of its frames
score below 0.50; 53 of its 68 misses are blind misses rather than
mislocalisations. The console orders it last and prints that figure next to it.

**Image-level labels are a simplification.** 12 of the 180 test frames carry more
than one defect type; the confusion matrix collapses those to a majority class.

**The economics are placeholders.** 12:1 miss-to-false-alarm and 5% prevalence
are defensible orders of magnitude, not Jindal Stainless numbers.

**This deployment specifically.** Free CPU tier: 2 shared vCPU, 16 GB RAM, no
GPU. It carries 18 of the 180 held-out test frames and one strip capture, not
the datasets. Uploads above 12 MP are downscaled before the network sees them
(a tiling cost, with the arithmetic in `app.py`). EigenCAM is behind a button
because the first call in a process costs seconds. Nothing is downloaded at
runtime; both checkpoints are committed. Everything the console imports lives in
`vendor/`, which is described in `vendor/README.md`.

---

## Files

    app.py                  the console, adapted from demo/app.py; deviations marked `# SPACE:`
    vendor/                 copies of the repository's src/ modules, with a drift checker
    models/                 both checkpoints, plus the gate's reference distribution
    reports/                the calibration map, the operating point, the capacity model
    data/neu-det/           18 held-out test frames, 3 per class
    assets/                 one 2048x1000 GC10-DET line-scan frame, CC BY 4.0
    tools/bench_cpu.py      produced every latency figure above
    tools/verify_vendor.py  diffs vendor/ against the repository's src/
    DEPLOY.md               how this Space was pushed
