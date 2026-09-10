# Clean-steel false alarm rate, and the operating point it implies

**Checkpoint** `yolov8n_neudet/best.pt` (`models/yolov8n_neudet/weights/best.pt`)
**Device** mps | **train imgsz** 320 | **NMS IoU** 0.45
**Clean patches mined from** `test` (106 images)
**Detection rate measured on** `val` (180 defective frames)
**Generated** 2026-09-09T03:54:48+00:00

## 0. The gap this closes

Every NEU-DET image contains a defect, so the false alarm rate in
`reports/evaluation.md` is the rate at which a **spurious extra box appears on a
coil that was already defective**. The coil was going to be flagged anyway. The
number a mill buys on is different: how often does the system stop the line for a
coil that is fine? That cannot be read off NEU-DET, so this report builds a proxy
and states what the proxy cannot tell you.

## 1. The proxy

771 crops were mined from the `test` split
that have **zero** intersection with any ground-truth box, with every crop grown by
6 px on all four sides before the test -- so a crop that
merely grazes a label is thrown away. Four crop scales
(60px, 80px, 100px, 120px) were used, candidates were
thinned to at most IoU 0.25 against each other and capped at
6 per image per scale, so no single image with a
large clean area can dominate the statistic. 106 of the
180 test images contributed at least one clean patch.

Each crop is run at imgsz = round_to_32(crop_px * deployment_magnification), where deployment_magnification = train_imgsz / frame_px. This keeps steel texture at the pixels-per-millimetre the network was trained on; running a crop at the full deployment imgsz would magnify it and measure a resampling artefact instead of the detector. Deployment magnification here is
1.60x (320 / 200), so the four
scales run at imgsz 96, 128, 160, 192
respectively -- achieved magnifications
1.60x, 1.60x, 1.60x, 1.60x.

## 2. False alarms on defect-free steel

At the recommended threshold **conf = 0.15**:

- **23.7%** of clean patches raise at
  least one box. **95% CI [17.7%,
  30.1%]**, from an image-clustered bootstrap over
  106 source images -- this is the interval
  to quote. The binomial Wilson interval
  [20.9%, 26.9%] is roughly
  half as wide and is wrong here: the 771 crops are not
  independent draws (design effect 4.3, effective
  n = 179, not 771).
- **0.27** boxes per clean patch
- if the false positives were spatially independent, that would extrapolate to
  **87.1%** of a full
  200x200 frame (best-fit Poisson intensity
  0.513 per 10,000 clean px). **The data rejects that model** -- see below,
  so this figure is a bound, not a prediction.

Per crop scale at that threshold. This is the evidence for or against extrapolating
a patch rate to a frame, and it is the most useful thing in this report:

| crop | area px | n | FA rate | 95% CI | boxes/patch | rate if independent |
|---|---|---|---|---|---|---|
| 60x60 | 3600 | 448 | 0.234 | [0.198, 0.276] | 0.245 | 0.169 |
| 80x80 | 6400 | 212 | 0.245 | [0.192, 0.307] | 0.302 | 0.280 |
| 100x100 | 10000 | 77 | 0.221 | [0.143, 0.325] | 0.299 | 0.401 |
| 120x120 | 14400 | 34 | 0.265 | [0.146, 0.431] | 0.412 | 0.522 |

The false alarm rate is **flat in crop area** over a 4x span of area. Fitting the
two competing one-parameter models to those four cells:

| model | what it says | chi2 | dof | p | chi2 (deff-corrected) | p (deff-corrected) |
|---|---|---|---|---|---|---|
| constant rate | false alarms are a property of the region, not of area | 0.35 | 3 | 0.950 | 0.20 | 0.978 |
| Poisson in area | false alarms are independent events per unit area | 34.56 | 3 | 1.5e-07 | 19.29 | 0.00024 |

Both chi-squares are computed against each model's **best** fit -- the Poisson
intensity is fitted by minimising the same statistic, not by an unweighted
least-squares linearisation, because the chi-square of a badly-estimated
parameter is evidence about the estimator, not about the model. The
design-effect-corrected columns divide by the mean per-cell design effect
(1.79), because the crops inside a cell
are clustered by source image and the nominal p-values assume they are not.

The constant-rate model fits (0.35 on 3 dof); the independence model does not (34.56 on 3 dof, 19.29 after the design-effect correction, p = 0.00024). Spatial independence is rejected -- and it stays rejected after the correction, which is the version to quote -- so the area extrapolation is **not** a prediction and is reported here only as a pessimistic bound.

The clustering probe says the same thing from the other direction. The
771 clean patches come from 106
source images, 7.3 patches each. At this threshold
52.8% of those images have at least one
alarming patch; if patches within an image were independent, that figure would be
73.9%.
The measured image-level rate is well below the independent prediction, so alarming patches clump inside a minority of images: some surface regions provoke the detector repeatedly and most never do.

**Practical consequence:** a false alarm on clean steel is a property of the local
surface texture, not a per-unit-area event. Doubling the inspected area does not
double the false alarms. The per-patch rate
(23.7%) is therefore the more defensible
frame-level estimate -- and it is itself an upper bound, for the reason in the next
subsection -- while the 87.1% extrapolation
should be read as a pessimistic bound rather than a prediction.


### The pooled rate is a mixture, and the weights are an artefact

A clean crop from a crazing coil and a clean crop from an inclusion coil are not
the same surface, and the mining does not sample them evenly -- how many clean
crops a defect class yields is set by how much of the frame its annotation covers.
Broken out by the defect class of the source image:

| source image defect class | clean crops | source images | FA rate | 95% Wilson |
|---|---|---|---|---|
| pitted_surface | 1 | 1 | 1.000 | [0.206, 1.000] |
| crazing | 41 | 12 | 0.488 | [0.343, 0.635] |
| scratches | 246 | 26 | 0.427 | [0.367, 0.489] |
| rolled-in_scale | 107 | 21 | 0.308 | [0.229, 0.401] |
| inclusion | 273 | 26 | 0.066 | [0.042, 0.102] |
| patches | 103 | 20 | 0.058 | [0.027, 0.121] |

Across the populations with at least 20 crops the rate runs from 5.8% (`patches`) to 48.8% (`crazing`) -- a spread of 8 times with non-overlapping intervals, so this is not noise. One class alone supplies 35% of all clean crops. The single pooled number therefore carries a composition chosen by the dataset's annotation habits, and a mill inspecting a different product mix would see a different figure from the same detector.

**And the boxes are not class-random.** **78%** of the boxes raised on
clean crops carry the source image's **own** defect class; a hallucination
unrelated to the coil would give 17%. That is the signature of the
detector firing on unlabelled continuation of the real defect -- crazing, scale
and scratches all extend past the boxed region in NEU-DET -- not of it inventing
defects on sound metal. It cannot be separated from genuine false alarming
without re-annotation, so **23.7% is an upper bound**, and the two
populations whose annotation is most plausibly exhaustive sit far below it.

Which defect does the model hallucinate on clean steel?

| class | false positives | share |
|---|---|---|
| scratches | 88 | 41.7% |
| pitted_surface | 40 | 19.0% |
| rolled-in_scale | 34 | 16.1% |
| crazing | 21 | 10.0% |
| inclusion | 19 | 9.0% |
| patches | 9 | 4.3% |

The single largest failure mode is **scratches** at 42% of all
false positives on clean metal. Scratches are a linear, low-contrast feature, and rolling, levelling and grinding leave linear texture on perfectly sound stainless; the model has never been shown that texture with a 'clean' label. Either way this is the confusion to
attack first, and the fix is hard negatives from clean strip -- exactly the data
this report says the project does not yet have.

## 3. Positive control -- is the small field of view silencing the model?

A low false alarm rate on small crops means nothing if the model simply stops
firing on small crops. 197 crops of the same four sizes that
**fully contain** a labelled defect (and clip no other box) were scored under the
identical protocol:

| conf | flag rate | localised detection rate (IoU 0.5, right class) |
|---|---|---|
| 0.01 | 1.000 | 0.949 |
| 0.05 | 0.990 | 0.924 |
| 0.15 | 0.980 | 0.909 |
| 0.25 | 0.924 | 0.868 |
| 0.40 | 0.787 | 0.746 |
| 0.60 | 0.538 | 0.523 |
| 0.80 | 0.279 | 0.279 |

At conf = 0.15 the control flags
98.0% of defect-containing crops and localises the
defect correctly on 90.9% of them,
against 23.7% on clean crops. The
protocol therefore separates defective from clean surface; the clean-patch number
is informative rather than an artefact of a starved field of view.

## 4. The re-derived operating point

Cost model, entirely in `COST_MODEL` at the top of `src/false_alarm.py` and open to
challenge:

    cost(t) = prevalence * ratio * (1 - D(t)) + (1 - prevalence) * F(t)

- `ratio` = 12 -- one escaped defect priced at
  12 unnecessary re-inspections
- `prevalence` = 0.05 -- fraction of inspected frames
  that actually carry a defect. **Not measurable from NEU-DET**, which is 100%
  defective by construction, and not public for Jindal (see
  `docs/research_notes.md` section 9). A placeholder.
- `D(t)` from real defective `val` frames, full 200x200 at
  imgsz 320
- `F(t)` from the clean-steel proxy, on the **patch** basis (the measured per-patch rate, with no extrapolation)

**Identifiability:** argmin of that cost is invariant to positive rescaling, so
prevalence and ratio move the recommendation only through the single weight
`lambda = prevalence * ratio / (1 - prevalence)` = 0.6316.
Arguing about the two separately is arguing about one number.

| conf | D(t) detect | clean patch FA | clean frame FA (extrap.) | legacy FA | cost (patch basis) | cost (frame basis) |
|---|---|---|---|---|---|---|
| 0.01 | 0.978 | 0.591 | 0.998 | 0.967 | 0.5752 | 0.9615 |
| 0.02 | 0.978 | 0.503 | 0.994 | 0.928 | 0.4914 | 0.9571 |
| 0.03 | 0.972 | 0.451 | 0.986 | 0.911 | 0.4455 | 0.9535 |
| 0.05 | 0.956 | 0.384 | 0.972 | 0.889 | 0.3914 | 0.9498 |
| 0.10 | 0.928 | 0.290 | 0.923 | 0.728 | 0.3193 | 0.9203 |
| 0.15 | 0.906 | 0.237 | 0.871 | 0.594 | 0.2822 | 0.8845 |
| 0.20 | 0.867 | 0.187 | 0.796 | 0.528 | 0.2574 | 0.8360 |
| 0.25 | 0.828 | 0.140 | 0.685 | 0.417 | 0.2364 | 0.7539 |
| 0.30 | 0.800 | 0.115 | 0.611 | 0.356 | 0.2297 | 0.7004 |
| 0.35 | 0.789 | 0.096 | 0.537 | 0.261 | 0.2178 | 0.6367 |
| 0.40 | 0.772 | 0.084 | 0.487 | 0.217 | 0.2168 | 0.5996 |
| 0.45 | 0.744 | 0.064 | 0.393 | 0.200 | 0.2137 | 0.5266 |
| 0.50 | 0.733 | 0.051 | 0.329 | 0.144 | 0.2081 | 0.4730 |
| 0.55 | 0.683 | 0.025 | 0.180 | 0.100 | 0.2134 | 0.3610 |
| 0.60 | 0.661 | 0.013 | 0.107 | 0.083 | 0.2157 | 0.3048 |
| 0.65 | 0.606 | 0.006 | 0.057 | 0.044 | 0.2428 | 0.2909 |
| 0.70 | 0.544 | 0.001 | 0.017 | 0.011 | 0.2746 | 0.2891 |
| 0.75 | 0.472 | 0.000 | 0.000 | 0.000 | 0.3167 | 0.3167 |
| 0.80 | 0.344 | 0.000 | 0.000 | 0.000 | 0.3933 | 0.3933 |
| 0.85 | 0.194 | 0.000 | 0.000 | 0.000 | 0.4833 | 0.4833 |
| 0.90 | 0.072 | 0.000 | 0.000 | 0.000 | 0.5567 | 0.5567 |
| 0.95 | 0.000 | 0.000 | 0.000 | 0.000 | 0.6000 | 0.6000 |

### Recommended threshold: **conf = 0.15**

At conf=0.15 the detector finds the real defect on 90.6% of defective frames while raising a box on 23.7% of steel crops carrying no annotated defect (95% image-clustered CI 17.7%-30.1%) (0.27 boxes per clean patch; an area extrapolation the data rejects would put this at 87.1% of full frames, but the measured rate is flat in crop area, so 23.7% is the better frame-level estimate). Pricing one escaped defect at 12 unnecessary re-inspections and assuming 5% of inspected frames are defective scores it at 0.2822 on the patch basis -- the lowest of every threshold clearing the 90% detection floor. On this basis the miss term is per frame and the false alarm term is per crop, so that score orders thresholds and is not a count of re-inspections per frame. For contrast, the legacy false alarm definition (a spurious box on an already-defective frame) reads 59.4% at the same threshold, which is higher than the clean-steel rate and measures a different event entirely. That pooled figure is a mixture: across the source defect classes with at least 20 clean crops it ranges from 5.8% to 48.8%, and the mixture weights are set by how much frame area each defect class leaves unannotated, not by anything about a mill. 78% of the boxes raised on clean crops carry the source image's own defect class (chance would be 17%), so an unknown share of them are unlabelled continuations of the real defect rather than hallucinations on sound metal: read 23.7% as an upper bound.

### Sensitivity to the cost ratio

Unconstrained = the cost minimum on economics alone; constrained = the cheapest
point that still clears the 90% detection floor.

| miss:FA | lambda | uncon. conf | D(t) there | clean FA there | clears floor | con. conf |
|---|---|---|---|---|---|---|
| 3:1 | 0.1579 | 0.60 | 0.661 | 0.013 | no | 0.15 |
| 10:1 | 0.5263 | 0.50 | 0.733 | 0.051 | no | 0.15 |
| 12:1 | 0.6316 | 0.50 | 0.733 | 0.051 | no | 0.15 |
| 30:1 | 1.5790 | 0.15 | 0.906 | 0.237 | yes | 0.15 |

The unconstrained optimum moves from conf 0.15 to 0.60 across a 10x span of the cost ratio -- a spread of 0.45. That is a wide swing: the recommendation is being set by the cost assumption at least as much as by the detector. 3 of the 4 ratios tested put the cost-minimising threshold somewhere that fails the 90% detection floor, so for those the floor -- not the economics -- fixes the answer at conf=0.15. The practical reading: the cost ratio only starts to matter once someone is willing to move the detection floor, and the number to settle with the quality department is therefore the acceptable escape rate first and the cost of an escape second.

### Basis check

Chosen on the raw per-patch false alarm rate: conf =
0.15. Chosen on the area-extrapolated per-frame rate:
conf = 0.15. The spatial-independence model behind the area extrapolation is rejected by the measured per-scale rates (chi2 34.556 vs 0.353 for a constant rate, 3 dof), so the primary basis here is the **patch** rate. Both bases pick the same threshold, so nothing in the recommendation rests on the extrapolation either way.

### Held-out detection cross-check

`D(t)` above is measured on `val`, and that is the split
ultralytics used to select `best.pt` (`models/*/args.yaml`: `val=true`,
`split=val`), so the detection half of the objective is not held out either. The
same sweep on `test`, which selected nothing, gives
**92.2%** at conf = 0.15
against 90.6% on the tuning split. The held-out figure is the higher of the two, so the recommendation's detection rate is not inflated by checkpoint selection; 90.6% is the conservative one to quote.

### Val cross-check

The clean patches above are mined from `test`, as the task specifies, so the false
alarm half of this objective is not held out. Repeating the whole mining and
scoring procedure on `val` gives 860 clean patches,
a false alarm rate of 19.7% at conf =
0.15 (test: 23.7%),
and a recommended threshold of 0.15. The two splits agree on the threshold, so the leak does not change the recommendation.

## 5. Chart

`false_alarm_curve.png` -- **A** false alarm rate against
detection rate for all three false-alarm definitions, with the detection floor and
the recommended point marked; **B** expected cost against threshold at each
sensitivity ratio, triangles at the unconstrained minima; **C** the area-dependence
test, measured per-scale rates with 95% Wilson bars against the constant-rate and
spatial-independence models.

## 6. What this proxy is not

1. The crops are smaller than a frame, so a per-patch rate is not a per-frame rate. The obvious bridge -- assume false positives are spatially independent and scale by area -- is tested here against the four measured crop scales rather than assumed, and on this checkpoint it is rejected; the per-frame figure is therefore carried only as a pessimistic bound.
2. The crops come from coils that do contain defects elsewhere. They are clean regions of defective coils, not clean coils.
3. A crop with no annotated box is not guaranteed defect-free: NEU-DET annotation is not exhaustive. Unlabelled faint crazing or light scale inside a 'clean' crop is scored here as a false alarm, which biases the measured rate upward. This is not a small effect: at the recommended threshold the large majority of boxes raised on 'clean' crops carry the source image's OWN defect class, far above the 1-in-6 a class-independent hallucination would give. The headline rate is therefore an upper bound on false alarming, and an unknown share of it is the detector finding real but unlabelled defect.
4. The pooled rate is a mixture over six surface populations that the mining samples very unevenly -- how many clean crops a defect class yields depends on how much of the frame its annotation covers, which has nothing to do with how often that surface appears on a line. The per-class rates differ by nearly an order of magnitude, so the single pooled figure carries a composition that is an artefact of the dataset, not of the mill.
5. The crops are not independent samples. Up to six are taken per image per scale, they overlap across scales, and alarming crops clump inside a minority of images. The binomial (Wilson) interval is therefore too narrow; an image-clustered bootstrap interval is reported and is the one to quote.
6. The field of view is small, so the model sees less context than it would on a full frame. The positive control quantifies how much of the detector's behaviour survives that, but it does not remove the difference.
7. The clean patches are mined from the test split, as specified. The false alarm half of the objective is therefore not held out from the threshold choice; a val-mined cross-check is reported so the size of that leak is visible.
8. Defect prevalence and the miss:false-alarm cost ratio are assumptions, not measurements, and they are not separately identifiable from the decision -- only their combination lambda = prevalence * ratio / (1 - prevalence) moves the recommended threshold.

### What a real measurement would need

Continuous capture of coils that passed manual surface inspection and shipped as prime, at production line speed, camera geometry and illumination, with frames time-stamped back to a coil id so a flagged frame can be adjudicated against the mill's own disposition record. A few thousand such frames across a shift, several coils, several grades and both strip surfaces would give a directly measured per-frame false alarm rate with a usable confidence interval.

Until that exists, the honest statement for a slide is:

> On unannotated stainless surface captured under the same imaging conditions, at
> conf = 0.15, the detector raises a box on **at most
> 24%** of
> 60-120 px clean patches
> (n = 771 crops from
> 106 images, 95% image-clustered CI
> 18%-30%), and that
> rate does not grow with inspected area. "At most", because **78%** of
> those boxes carry the source image's own defect class, so some are unlabelled
> real defect. Over the same threshold it detects the real defect on
> 91% of defective frames
> (92% on the fully held-out split).
> This is a proxy measured on clean regions of defective coils, not a false alarm
> rate on clean coils; the latter has not been measured and cannot be measured
> from NEU-DET.

Three numbers to refuse to put on a slide, because this work cannot support them:
a false alarm rate per coil, a false alarm rate per kilometre of strip, and any
statement about how often the line would actually stop. All three need the
production capture described above.
