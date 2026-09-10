# Photometric normalisation and the out-of-distribution gate

Closes audit gap 1 part C (`docs/audit/technical_headroom.md` §1, photometrics) and the
backend of gap 2 (`docs/audit/demo_ux.md` finding 1: *"anything that is not steel is
confidently dispositioned as a critical defect"*).

Two modules, both pure numpy/OpenCV, neither importing torch:

| module | question it answers |
|---|---|
| `src/domain_shift.py` | the frame is steel, but from a different camera — what should the pixels look like? |
| `src/ood_guard.py` | is this a frame of steel at all? |

Machine-readable companion: `reports/ood_guard.json`.
Everything below was executed on this working tree on 2026-09-09 against
`models/yolov8n_neudet/weights/best.pt` at imgsz 256, Apple M5 / MPS.

**The headline is untouched.** Re-verified after both modules landed:

```
mAP50=0.7524 mAP50-95=0.3967 P=0.6960 R=0.6867      (split=test, imgsz=256, device=mps)
```

Normalisation is **off by default** in the deployed pipeline, and the gate is read-only and
passes 180/180 test frames, so neither module can move that number. Both facts are locked by
tests (`test_the_gate_passes_every_frame_of_the_headline_test_split`).

---

## Part 1 — `src/domain_shift.py`

### The reference

`models/domain_reference.json` (schema `jsw-domain-reference/2`), built from the NEU-DET
**training** split only — never val or test, because the reference ships with the model and
may only have seen what the model saw. It ships next to the weights so a deployed pipeline
never needs 1,440 JPEGs on disk.

| | measured here | audit said |
|---|---|---|
| NEU-DET train, pooled grey | **mean 129.2, σ 54.3** (1,440 frames, 57.6 M px) | mean 130.2, σ 41.1 |
| NEU-DET train, typical single frame | mean 129.1, σ **30.1** | — |
| NEU-DET train, mean of per-frame σ | 26.8 | (this is what σ 41.1 was probably measuring) |
| NEU-DET train, σ of per-frame means | 44.6 | — |
| Severstal, pooled grey (1,400 real frames) | **mean 87.3**, σ 49.4 | mean 86.5 |

The means reproduce to within a grey level. The σ discrepancy is the interesting part and it
changed the design: **the pooled σ of 54.3 is inflated by exposure differences *between*
frames** (σ of frame means = 44.6), so it is not what a single frame looks like. The module
therefore stores two targets — `population` (every pixel pooled) and `frame` (per-frame
histograms averaged after each is shifted onto the population mean, σ 30.1) — and matching
onto the wrong one over-stretches contrast. Which target you pick turns out to matter more
than the method.

### Reproducing the audit's result

> *"Histogram-matching the clean crops to NEU-DET cuts crop FA from 44.1 % → 33.6 % at
> conf 0.25 (−24 % relative) for one line of preprocessing."*

Measured on 3,200 crops of genuinely defect-free real Severstal strip and 1,079
mask-confirmed defect crops:

Four variants, all onto the pooled population target at full strength. "Scope" is what the LUT
is derived from — the whole 1600×256 frame, or each 200×200 crop independently.

| variant | clean-crop FA @0.25 | defect-crop recall @0.25 |
|---|---|---|
| baseline (no normalisation) | 44.3 % | 58.3 % |
| histogram match, **frame** scope | **34.1 %** | 43.1 % |
| histogram match, crop scope | 36.4 % | 43.8 % |
| affine (mean/σ only), crop scope | 39.6 % | 46.0 % |

**The false-alarm number reproduces.** 44.3 → 34.1 against the audit's 44.1 → 33.6.

**And it is not a win.** Recall fell with it, 58.3 % → 43.1 %. The threshold no longer means
what it meant. Held at *equal recall*, every variant is worse than doing nothing:

| variant | AUC | conf at matched 58.3 % recall | clean-crop FA at that point |
|---|---|---|---|
| baseline | **0.615** | 0.250 | **44.3 %** |
| histogram, frame scope | 0.593 | 0.152 | 45.8 % |
| histogram, crop scope | 0.566 | 0.164 | 49.4 % |
| affine (mean/σ only), crop scope | 0.556 | 0.177 | 50.5 % |

The entire apparent gain was a threshold artefact. This is the single most important
correction in this report: **the audit's one-line win, taken literally, makes the system
worse.**

### What actually helps

Sweeping both **targets** against blend strength, all at frame scope (`strength` blends the
LUT towards the identity, not the pixels, so the mapping stays monotone). Note *target* and
*scope* are different axes: scope is what the LUT is derived **from**, target is the stored
distribution it is mapped **onto**. Row `population / 1.00` is the audit's recipe, and matches
the `histogram match, frame scope` row above exactly.

| target | strength | NEU-DET test mAP50 | cross-domain AUC | clean FA at matched 58.3 % recall |
|---|---|---|---|---|
| — | 0.00 (off) | **0.7533** | 0.615 | 44.3 % |
| population | 0.25 | 0.7249 | 0.629 | 40.1 % |
| population | 0.50 | 0.6503 | 0.614 | 43.0 % |
| population | 1.00 | 0.4912 | 0.593 | 45.8 % |
| **frame** | **0.25** | **0.7242** | **0.641** | **38.0 %** |
| frame | 0.50 | 0.6483 | 0.634 | 38.8 % |
| frame | 0.75 | 0.5548 | 0.631 | 38.6 % |
| frame | 1.00 | 0.4417 | 0.624 | 38.9 % |

The typical-frame target at strength 0.25 — the module defaults — is a **real** gain, because
it moves AUC rather than just moving the threshold: **0.615 → 0.641**, and at matched recall
the clean-strip false-alarm rate goes **44.3 % → 38.0 %** (−14 % relative).

*(The 0.7533 in row 1 is the re-encode control: decode → re-save → score, with no mapping
applied. It differs from the 0.7524 headline by +0.0009, which is JPEG round-trip noise and
bounds the precision of every mAP50 comparison in this table.)*

### What it costs in-domain — the reason it ships off

NEU-DET held-out test, shipped weights, one preprocessing change:

| configuration | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| **shipped, no normalisation** | **0.7524** | **0.3967** | 0.6960 | 0.6867 |
| re-encoded, no mapping (control) | 0.7533 | 0.3950 | 0.7001 | 0.6916 |
| histogram, frame target, strength 0.25 | 0.7242 | 0.3557 | — | — |
| histogram, population target, strength 0.50 | 0.6503 | 0.3089 | 0.6042 | 0.6125 |
| histogram, population target, strength 1.00 | 0.4912 | 0.2043 | 0.5076 | 0.5154 |
| affine, population target, strength 1.00 | 0.5662 | 0.2725 | 0.5049 | 0.5663 |

mAP50 falls monotonically with strength. The cause is not subtle: **the shipped weights were
fitted on un-normalised frames, so any per-frame remapping is train/serve skew.**

So `domain_shift` is **off by default in the deployed pipeline** and the 0.7524 headline
stands. It is a **per-source commissioning setting** — switched on for a camera that is not
the NEU-DET rig — not a per-frame decision, because `histogram_distance` provably cannot tell
the two populations apart (see the next section). The way to collect the cross-domain benefit
without the in-domain cost is to normalise the *training* data and re-fit; that is a training
change and out of this module's scope.

### API

```python
from domain_shift import normalise_frame, match_to_reference, load_reference

frame, record = normalise_frame(image)          # frame target, strength 0.25
```

On a real Severstal frame (`006a4402e.jpg`), `record["note"]` is verbatim:

```
Photometric normalisation: frame grey mean 87 sigma 41 -> 98 sigma 38, matched to the
NEU-DET training distribution (mean 129 sigma 54). Distance from that distribution
43 -> 32 grey levels.
```

`match_to_reference(image, method=…, target=…, strength=…)` returns RGB uint8;
`method="none"` and `strength=0.0` are validated bit-identical no-ops so a caller can A/B
without branching. The LUT is monotone non-decreasing at every strength (tested over both
targets × four strengths), which is the safety argument: normalisation can move a defect's
contrast against its background but can never invert it.

---

## Part 2 — `src/ood_guard.py`

### What the audit found, reproduced here

The detector has no seventh class for "not steel", so it answers everything. Reproduced on
this tree, at the shipped operating point:

| input | model verdict |
|---|---|
| plain white 400×400 frame | `pitted_surface 0.74`, **severity 86.3 critical** |
| cartoon landscape | 5 detections, `patches 0.89`, **severity 96.2 critical** |
| greyscale company logo | 2 detections, `inclusion 0.40`, severity 74.1 high |
| 8×8 px thumbnail | 2 detections, `inclusion 0.35`, **severity 93.5 critical** |
| flat grey / Gaussian noise | **0 detections** |

The audit's fairness control holds: it is not "fires on anything". It fires on **unfamiliar
steel-*like* texture**.

### The gate

Six checks. Each is one number against one threshold. Reject if any fails; the failing checks
*are* the explanation. Nothing is trained and nothing is fitted.

| check | statistic | reject when | worst value on 3,200 real steel frames | margin |
|---|---|---|---|---|
| resolution | shorter side | < 64 px | 200 px (NEU-DET) | 3.1× |
| aspect | longest : shortest | > 64 : 1 | 6.25 : 1 (Severstal) | 10.2× |
| colour | mean HSV saturation | ≥ 40 / 255 | **0.00** on all 3,200 | ∞ |
| noise | fraction of px with \|Laplacian\| > 32 | ≥ 0.70 | 0.538 | 1.30× |
| focus | Laplacian variance, luminance downscaled to 256 px | < 1.5 | 7.76 (Severstal) | 5.2× |
| tonal range | entropy of the **unclipped** grey histogram | < 1.8 bits | 2.08 bits | 1.16× |

Two of these deserve their reasoning stated:

**Tonal range excludes clipped pixels, and that is the whole trick.** A 92 %-blown-out but
genuine Severstal frame and a white-background logo sit at 1.75 vs 1.60 bits on the raw
histogram — a 9 % gap, unusable. Dropping the clipped mass (levels ≤2 and ≥253, which carry no
surface information) moves them to 2.08 vs 1.60, a 30 % gap. Quoted to an operator as
`2 ** bits` = *effective grey levels*: genuine strip runs at 70–150, a printed logo at 3.0.

**Focus is measured at 256 px, not native.** Scale-normalising to `DEFAULT_IMGSZ` is what makes
a 4× upscale of a soft frame read as soft rather than sharp.

**One signal is reported but forbidden from gating.** Distance from the NEU-DET training grey
histogram cannot separate: real steel spans 4.3–126.1 grey levels of distance, and a blank
white frame scores 125.8 — *inside* that range. It instead drives the third decision,
`REVIEW`: a frame that passes all six checks but sits past the training split's own maximum
(116.5) is scored *and* flagged, which is exactly the case `domain_shift` exists for.

### Measured — the confusion table

**Genuine steel. Every rejection here would be a false rejection.**

| suite | n | pass | review | **reject** |
|---|---|---|---|---|
| NEU-DET train | 1,440 | 1,440 | 0 | **0** |
| NEU-DET val | 180 | 179 | 1 | **0** |
| NEU-DET test (the headline split) | 180 | 180 | 0 | **0** |
| Severstal real strip frames | 1,400 | 1,350 | 50 | **0** |
| **total** | **3,200** | 3,149 | 51 | **0** |

**Zero false rejections on 3,200 genuine steel frames.** The 51 `REVIEW`s are all exposure
flags, not refusals — those frames are still scored.

**Negatives — the tuning suite (thresholds were chosen against these, so this table is
descriptive, not evidence):**

| case | gate | caught by | what the model would have said |
|---|---|---|---|
| solid_white | REJECT | focus, tonal_range | `pitted_surface 0.74`, sev 86.3 **critical** |
| solid_black | REJECT | focus, tonal_range | `pitted_surface 0.34`, sev 66.4 high |
| solid_grey130 | REJECT | focus, tonal_range | 0 det |
| solid_red | REJECT | colour, focus, tonal_range | 0 det |
| solid_blue | REJECT | colour, focus, tonal_range | 0 det |
| linear_gradient | REJECT | focus | `pitted_surface 0.27`, sev 63.2 high |
| gauss_noise_grey | REJECT | noise | 0 det |
| gauss_noise_colour | REJECT | colour, noise | 0 det |
| synthetic_colour_photo | REJECT | colour, noise | 0 det |
| cartoon_scene | REJECT | colour, tonal_range | 5 det, `patches 0.89`, sev 96.2 **critical** |
| logo_colour | REJECT | tonal_range | 2 det, `inclusion 0.19`, sev 65.2 high |
| logo_greyscale | REJECT | tonal_range | 2 det, `inclusion 0.40`, sev 74.1 high |
| screenshot_text | REJECT | tonal_range | `pitted_surface 0.40`, sev 69.3 high |
| blurred_steel_σ3 | REJECT | focus | 3 det, `inclusion 0.55`, sev 100.0 **critical** |
| blurred_steel_σ6 | REJECT | focus | 2 det, `inclusion 0.49`, sev 98.5 **critical** |
| tiny_8x8 | REJECT | resolution | 2 det, `inclusion 0.35`, sev 93.5 **critical** |
| tiny_40x40 | REJECT | resolution | 4 det, `inclusion 0.52`, sev 99.6 **critical** |
| strip_6000x10 | REJECT | resolution, aspect, focus | **detector crashes** (`cv2.error` in letterbox) |

**18 / 18.** Note the last row: the gate also catches an input the detector cannot process at
all — a 6000×10 frame raises `cv2.error: inv_scale_x > 0` inside ultralytics' letterbox.

**Negatives — held out.** Built *after* the thresholds were frozen, with a different seed and
different constructions. No threshold was changed in response to this table. **This is the
evidence.**

| case | gate | caught by | what the model would have said |
|---|---|---|---|
| text_document | REJECT | tonal_range | `pitted_surface 0.16`, sev 44.7 medium |
| spreadsheet | REJECT | tonal_range | 8 det, `scratches 0.39`, sev 68.9 high |
| bar_chart | REJECT | colour, tonal_range | 4 det, `scratches 0.35`, sev 74.3 high |
| qr_code | REJECT | tonal_range | 7 det, `inclusion 0.46`, sev 85.8 **critical** |
| barcode | REJECT | tonal_range | 5 det, `scratches 0.77`, sev 93.8 **critical** |
| cartoon_face | REJECT | tonal_range | 5 det, `inclusion 0.81`, sev 92.9 **critical** |
| checkerboard | REJECT | tonal_range | 0 det |
| colour_mesh | REJECT | colour | 5 det, `inclusion 0.44`, sev 86.2 **critical** |
| landscape_photo | REJECT | colour, noise | `patches 0.86`, sev 41.6 medium |
| defocused_scene | REJECT | colour | `patches 0.67`, sev 38.8 medium |
| uniform_noise | REJECT | noise | 0 det |
| solid_grey64 | REJECT | focus, tonal_range | 0 det |
| tiny_32x32 | REJECT | resolution | 4 det, `patches 0.69`, sev 89.0 **critical** |
| **radial_gradient** | **PASS** | — | `pitted_surface 0.39`, sev 69.0 high |
| **wood_grain** | **PASS** | — | `pitted_surface 0.63`, sev 80.7 **critical** |

**13 / 15 on held-out negatives, with no retuning.**

**Hard positives — genuine steel, degraded. All 11 pass:** plain NEU-DET frame, real Severstal
frame, lighting gradient, 15-tap motion blur, JPEG q20, under-exposed 0.5×, over-exposed 1.7×,
upscaled 4× and 16×, 5 % salt-and-pepper, and a warm colour cast (saturation 28.9/255, i.e.
1.4× under the colour threshold — the tightest margin any hard positive has).

### What it misses, and why that is structural

Both held-out misses are **greyscale, in focus, and textured**: a sinusoidal wood grain and a
radial gradient. The noise check's own margins are the next-thinnest thing in the module —
Gaussian grey noise clears the 0.70 threshold by 1.28×, uniform noise by 1.34×.

This is the honest limit, and it is not fixable by moving a threshold: **the gate tests
physics, not semantics.** Anything monochrome, focused and textured is indistinguishable from
rough strip on these six signals. Closing it needs a texture model or real negatives — that
is `docs/audit/technical_headroom.md`'s Severstal joint-training result, not another constant.

Two things make the residual risk smaller than the table suggests:

1. The model returns **0 detections** on several of the cases the gate misses or barely
   catches (grey noise, uniform noise, checkerboard) — gate and model fail in *different*
   places, so the combined system is stronger than either.
2. `wood_grain` at severity 80.7 critical is a genuine uncaught failure. It is asserted as a
   characterisation test (`test_a_focused_greyscale_texture_photograph_is_a_documented_miss`)
   so it cannot silently change without this report being updated.

### A design decision worth recording

The obvious optimisation — compute every statistic on the 256 px plane the detector sees,
instead of at full resolution — was measured and **rejected**:

| statistic | separation at full res | separation on the 256 px plane |
|---|---|---|
| noise (real max vs noisiest negative) | 1.56× | 1.21× |
| tonal range (real min vs flattest negative) | **1.30×** | **0.38×** |
| colour (quietest colour negative) | 117.9 | 67.8 |

A downscaled QR code gains antialiased grey levels and would pass. Only `focus` is measured on
the downscaled plane, where scale-normalisation is the point.

### Cost

Best of 5 runs of 300 (60 for the detector), warm:

| | best | median |
|---|---|---|
| gate, 200×200 frame | **0.277 ms** | 0.287 ms |
| gate, 1600×256 frame | 3.895 ms | 3.978 ms |
| `detector.predict`, 200×200 | 9.822 ms | 12.030 ms |

**~2.8 % overhead** on the frame size the console actually shows. The Laplacian uses `CV_16S`, which
is bit-identical to `CV_64F` for a thresholded count on a uint8 plane (verified: 0 of 633
frames differ) at 1.8× the speed.

### One caveat the frame-level table hides

On **256×256 crops** cut from the Severstal strip (1,200 crops), the gate rejects 133. Of
those, **130 are blank** — crops falling off the edge of the strip, carrying no steel
(rejected-crop grey σ: p95 = 1.24, against p05 = 10.68 for kept crops). Rejecting them is
correct. **3 are arguable false rejections** (0.25 %): two heavily-clipped crops at 1.25 and
1.53 tonal bits, and one very rough crop at edge density 0.732. A tiled pipeline should
therefore treat a rejected tile as *"no evidence here"* and carry on, not as a frame-level
refusal.

### API for the console

```python
from ood_guard import inspect

verdict = inspect(frame)                 # before normalisation, on the raw frame
if not verdict.ok:
    banner(verdict.headline, verdict.reason)     # and do NOT run the model
else:
    result = detector.predict(frame)
    if verdict.review:
        note(verdict.reason)             # scored, but outside the trained exposure range
```

`verdict.decision` ∈ `{"pass", "review", "reject"}`; `.ok` is `decision != "reject"`.
`.checks` is the full six-row table (name, value, threshold, passed, units, explanation, and a
`.margin` ratio) for a diagnostics panel. `.to_dict()` is JSON-serialisable.

Real CLI output:

```
$ .venv/bin/python src/ood_guard.py .../logo_greyscale.png
REJECT  .../logo_greyscale.png
        The detector was not run: the unclipped image uses 3.0 effective grey levels
        (1.60 bits, floor 1.8); printed graphics, logos and screenshots look like this,
        steel does not. A six-class defect model has no 'not steel' answer, so any
        verdict on this image would be meaningless rather than wrong.
        [ok  ] resolution      260.000 >= 64       (px)
        [ok  ] aspect            2.000 <= 64       (:1)
        [ok  ] colour            0.000 <= 40       (/255)
        [ok  ] noise             0.044 <= 0.7      (fraction)
        [ok  ] focus          2098.484 >= 1.5      (Laplacian variance)
        [FAIL] tonal_range       1.596 >= 1.8      (bits)
        [    ] hist_distance     94.0 (reported only)
```

### Commissioning a new camera

```
$ .venv/bin/python src/ood_guard.py --calibrate data/neu-det/test/images
calibration over 180 frames: 0 rejected, 0 review
  resolution   threshold 64       worst    200.000 margin   3.12x  failed 0
  aspect       threshold 64       worst      1.000 margin  64.00x  failed 0
  colour       threshold 40       worst      0.000 margin    infx  failed 0
  noise        threshold 0.7      worst      0.531 margin   1.32x  failed 0
  focus        threshold 1.5      worst     10.352 margin   6.90x  failed 0
  tonal_range  threshold 1.8      worst      4.755 margin   2.64x  failed 0
```

Point `--calibrate` at a few hundred known-good frames from a new line. Any check with
`worst_margin` near 1.0 will start rejecting real product. `suggest_thresholds()` backs the
four texture thresholds off by a safety factor and **only ever loosens** them — rejecting real
steel is the failure mode this module exists to avoid, so calibration is never allowed to make
it likelier (tested).

---

## Honest limitations

1. **The tuning negatives were used to pick the thresholds.** The 18/18 table is descriptive.
   The 13/15 held-out table is the evidence.
2. **Every negative is synthetic.** No photograph of a real object was available offline. The
   *positive* side — 3,200 frames from two real steel datasets, two different mills, two
   different cameras — is real, and that is the side where a mistake costs product.
3. **The colour check has never seen a real colour mill camera.** Both datasets are stored
   greyscale (saturation exactly 0.00 on all 3,200 frames), so the 40/255 threshold is
   validated only against a synthetic warm cast at 28.9. This is the first threshold to
   re-calibrate on a real installation.
4. **The blur gate has a genuine grey zone.** A 15-tap motion blur of real steel sits at
   focus 2.3 and passes; a σ-3 Gaussian blur sits at 1.1 and is rejected. Between 1.1 and 2.3
   the gate's answer is a judgement call, not a measurement.
5. **`domain_shift`'s cross-domain gain is measured on Severstal only** — one other dataset,
   crop-level, at one operating point. It is evidence that photometric normalisation helps
   across a camera change; it is not a plant-validated number.
6. **The gate cannot detect the failure it most needs to.** It catches images that are not
   photographs of surfaces. It does not catch a photograph of the *wrong surface* — aluminium,
   a conveyor belt, a concrete floor — which would pass all six checks. That is the same
   texture-model gap as the two held-out misses.

## Reproducing

```bash
.venv/bin/python src/domain_shift.py                       # print the cached reference
.venv/bin/python src/domain_shift.py --rebuild             # recompute from the training split
.venv/bin/python src/ood_guard.py FRAME [FRAME ...]        # inspect frames
.venv/bin/python src/ood_guard.py --calibrate DIR          # headroom on known-good frames
.venv/bin/python -m pytest tests/test_smoke.py -q          # 149 tests, incl. 38 for these modules
```
