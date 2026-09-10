# GC10-DET checkpoint: what it scores, and what that does and does not cover

**Weights** `models/yolov8n_gc10_640/weights/best.pt` (epoch 53 of 70, selected by
val mAP50)
**Evaluated** 2026-09-10, on the held-out GC10 **test** split only (333 images,
530 boxes), never used for training or model selection
**Dataset** `data/gc10-det/`, built and documented by `reports/gc10_dataset.md`
**Machine-readable** `reports/gc10_coverage.json`

This report answers one question honestly: *does the roll-mark and edge-defect
coverage this project now has actually work?* Short answer -- roll marks
(`rolled_pit`, `crease`) are trained but weak and thin; the edge-family proxy
(`crescent_gap`, `waist_fold`) is trained and strong, but it is a proxy for edge
geometry, not for cracks, and no crack class exists anywhere in this pipeline.

---

## Verdict

| question | answer |
|---|---|
| Does `models/yolov8n_gc10_640/weights/best.pt` load and match its claimed spec? | **Yes.** 10 classes, correct names and order, task=detect. Training args confirm 70/70 epochs, batch 16, imgsz 640, seed 1337. Section 1. |
| Do the dataset counts, class mapping, and split-disjointness claims in `reports/gc10_dataset.md` hold up under independent recount? | **Yes, exactly.** Every per-split box count, the class-id mapping, and the zero-overlap disjointness claim (by content hash and by line-scan sequence id) were recomputed from disk from scratch and matched the report to the box. Section 1. |
| What is the overall held-out test score? | **mAP50 0.5821, mAP50-95 0.3076**, precision 0.592, recall 0.590, on 333 images / 530 boxes. Section 2. |
| How well are roll marks (`rolled_pit`, `crease`) covered? | **Weakly, and the number is thin.** Mean AP50 0.20 across 24 test instances (rolled_pit 13, crease 11). A live sweep on all 11 rolled_pit test images shows 4 of 11 produce **zero** detections at conf 0.15, and most surviving detections sit at confidence 0.2-0.5. Section 3. |
| How well is the edge family (`crescent_gap`, `waist_fold`) covered? | **Well, as a proxy.** Mean AP50 0.88 across 61 test instances (crescent_gap 40, waist_fold 21). But these are edge-region geometry defects (a scalloped bite, a folded-over edge), **not cracks** -- this checkpoint has never seen a crack of any kind. `waist_fold`'s number is partly inflated by very large boxes. Section 3. |
| Do real predictions on real images look sane? | **Mostly yes, with one clean, representative miss.** 4 hand-picked test images: 3 land close to ground truth (crescent_gap, crease, waist_fold); the 4th shows the model missing a rolled_pit box entirely while correctly catching a co-occurring weld_line in the same frame -- and that miss is representative, not cherry-picked (Section 4's 11-image sweep). |
| Does the brief's "edge cracks" family now have a detector? | **No.** It has a stated, measured proxy. Saying otherwise would misrepresent what was trained. Section 5. |
| Was anything retrained or changed outside this report? | **No.** `resolve_weights()`, the shipped model, the joint model, and the demo are untouched. This checkpoint stands alone. |

---

## 1. Verification: model and dataset match their claims

### 1.1 Model loads

```
$ .venv/bin/python -c "from ultralytics import YOLO; m = YOLO('models/yolov8n_gc10_640/weights/best.pt'); print(m.names); print(m.task)"
names: {0: 'punching_hole', 1: 'weld_line', 2: 'crescent_gap', 3: 'water_spot', 4: 'oil_spot',
        5: 'silk_spot', 6: 'foreign_object', 7: 'rolled_pit', 8: 'crease', 9: 'waist_fold'}
task: detect
```

10 classes, correct names, correct order (matches `data/gc10-det/data.yaml`).

### 1.2 Training actually finished, at the claimed settings

`models/yolov8n_gc10_640/args.yaml`: `epochs: 70`, `batch: 16`, `imgsz: 640`,
`seed: 1337`, `data: data/gc10-det/data.yaml`.

`models/yolov8n_gc10_640/results.csv` has 70 logged rows. Best val mAP50 is
**0.55058 at epoch 53** (recomputed from the CSV, not read from a claim) --
matches the stated 0.5506. The final epoch-70 val mAP50 is 0.54936, essentially
identical, so `best.pt` (the epoch-53 checkpoint ultralytics saves) is not
masking a collapse later in training.

### 1.3 Dataset counts, independently recounted from the label files on disk

Recount script: parse every `data/gc10-det/{train,val,test}/labels/*.txt`
directly (not `manifest.json`).

| split | images | label files | boxes |
|---|---|---|---|
| train | 1,641 | 1,641 | 2,503 |
| val | 320 | 320 | 531 |
| test | 333 | 333 | 530 |

Per-class box counts for all 10 classes across all 3 splits were recomputed and
matched `reports/gc10_dataset.md` section 4 exactly, box for box, including the
brief-relevant classes:

| class | train | val | test |
|---|---|---|---|
| rolled_pit | 59 | 13 | 13 |
| crease | 52 | 11 | 11 |
| crescent_gap | 185 | 40 | 40 |
| waist_fold | 101 | 22 | 21 |

### 1.4 Split disjointness, independently re-verified

Recomputed from scratch: SHA-256 content hash of every image file, plus a
line-scan sequence id parsed straight from the filename
(`img_<camera>_<sequence>_<frame>`), not read from the manifest.

| check | result |
|---|---|
| unique image content hashes | 2,294 (= total image count; zero duplicates anywhere) |
| cross-split content-hash overlaps (train/val, train/test, val/test) | 0, 0, 0 |
| unique line-scan sequences per split | 252 / 108 / 107 (467 total) |
| cross-split sequence overlaps (train/val, train/test, val/test) | 0, 0, 0 |

This matches `reports/gc10_dataset.md` section 3.3 exactly. **No leakage found,
confirmed independently.**

**Verdict on Section 1: every claim in `reports/gc10_dataset.md` about the model
and dataset held up. Nothing needed correcting.**

---

## 2. Held-out test evaluation

```
$ .venv/bin/yolo detect val \
    model=models/yolov8n_gc10_640/weights/best.pt \
    data=data/gc10-det/data.yaml \
    split=test imgsz=640 batch=16 device=mps conf=0.001 iou=0.7

Ultralytics 8.4.144, torch 2.14.0, MPS (Apple M5)
                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                     all        333        530      0.592       0.59      0.582      0.308
           punching_hole         50         50      0.882          1      0.993      0.608
               weld_line         77         77      0.745      0.874      0.892      0.473
            crescent_gap         40         40      0.812      0.969      0.923      0.617
              water_spot         47         53      0.572      0.774      0.781      0.448
                oil_spot         42         84      0.577      0.358      0.428      0.192
               silk_spot         89        129      0.488      0.295      0.292      0.109
          foreign_object         37         52      0.409       0.25       0.27     0.0889
              rolled_pit         11         13      0.211      0.231      0.114     0.0505
                  crease         11         11      0.575      0.364      0.292       0.14
              waist_fold         21         21      0.646      0.783      0.837      0.348
Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 3.1ms postprocess per image
```

Reproduced three times: this `yolo` CLI invocation, the `ultralytics.YOLO(...).val(...)`
Python API, and a second independent session re-running both -- identical numbers every
time. **`iou=0.7` is not a free choice here**: it is the "ultralytics validator defaults"
value that `reports/evaluation.json`'s protocol field names, and the same value every other
val run in this project uses (unspecified, so ultralytics' own default) for the shipped,
joint, and 640px-experiment checkpoints. An earlier draft of this report used `iou=0.6`
without stating why, which moved overall mAP50 from 0.582 to 0.592 -- a real, if small,
effect, not noise, and not consistent with how every other headline number in this project
was produced. It is corrected here to match project convention.

**Overall: mAP50 0.5821, mAP50-95 0.3076, precision 0.5916, recall 0.5897** on
333 test images / 530 test boxes.

This is a 10-class head trained on line-scan strip frames 8x wider than
NEU-DET's crops; it is not directly comparable to the shipped model's 0.7524 on
NEU-DET's 6-class, tightly-cropped test set, and this report does not claim it
is. Table above gives instance counts next to every number for exactly that
reason -- `silk_spot` at 129 instances and `rolled_pit` at 13 instances are not
equally trustworthy AP values even though they are printed with the same number
of decimal places.

---

## 3. The classes the brief actually asked about

The brief names four families: scratches, scale, roll marks, edge cracks.
NEU-DET already covers the first two (`scratches`, `rolled-in_scale`). GC10-DET
was ingested specifically to close the last two. Here is what that ingest
actually bought, in test-set numbers.

### 3.1 Roll marks -- `rolled_pit`, `crease` (direct match, thin)

| class | test images | test instances | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| rolled_pit | 11 | 13 | 0.211 | 0.231 | **0.114** | 0.051 |
| crease | 11 | 11 | 0.575 | 0.364 | **0.292** | 0.141 |
| **mean (roll marks)** | | **24** | 0.393 | 0.297 | **0.203** | 0.096 |

**This is weak, and the weakness is confirmed by actually running the model,
not just by reading the AP number.** A sweep of all 11 GC10 test images that
contain a `rolled_pit` box, at conf 0.15:

| outcome | count |
|---|---|
| image produces zero `rolled_pit` detections | 4 / 11 |
| image produces at least one `rolled_pit` detection | 7 / 11 |
| confidence range on the detections that do appear | 0.16 - 0.71 (median approx 0.34) |

Box counts do not reliably match either: one image with 1 ground-truth box gets
2 predictions, another with 2 ground-truth boxes gets 3. This is a genuinely
weak, low-confidence class in practice, not an artefact of how AP is computed.

**How confident is the 0.114 / 0.292 number?** Not very, and it should not be
quoted more precisely than that. rolled_pit has 13 test instances: one instance
flipping from miss to hit moves recall by **1/13 = 7.7 points**. crease has 11:
one flip moves recall by **1/11 = 9.1 points**. The project's own bootstrap
half-width on the shipped model's 446-box NEU-DET test set was +/-0.025; nothing
computed on 11-13 boxes can be narrower than the effect it is trying to measure.
**Report these two numbers with their instance counts attached, or not at all.**

### 3.2 Edge family -- `crescent_gap`, `waist_fold` (proxy, not cracks)

| class | test images | test instances | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| crescent_gap | 40 | 40 | 0.812 | 0.969 | **0.923** | 0.617 |
| waist_fold | 21 | 21 | 0.646 | 0.783 | **0.837** | 0.348 |
| **mean (edge family)** | | **61** | 0.729 | 0.876 | **0.880** | 0.483 |

This is a strong, credible number on a reasonable instance count (61, versus
24 for roll marks) -- and it is still a proxy. **`crescent_gap` is a scalloped
bite out of the strip edge; `waist_fold` is a folded-over edge.** Neither is a
crack. This checkpoint has never been shown a labelled crack and detecting one
is not something it can be claimed to do.

One further caveat specific to `waist_fold`, carried over from
`reports/gc10_dataset.md` section 6 and worth restating here because it directly
affects how the 0.837 should be read: `waist_fold`'s median ground-truth box
covers **33.8% of the frame area**. A box that large is easy to overlap at
IoU 0.5 almost by construction, so part of that AP50 reflects box size rather
than localisation precision. `crescent_gap`, at a 5.3% median box area, is the
more meaningful of the two numbers.

### 3.3 Reading the two families side by side

| family | GC10 classes | match quality | test instances | mean AP50 |
|---|---|---|---|---|
| roll marks | rolled_pit (13), crease (11) | direct | 24 | 0.203 |
| edge cracks (brief) | crescent_gap (40), waist_fold (21) | **proxy only** | 61 | 0.880 |

The honest summary: the family with real matching data (roll marks) is weak and
thin; the family with strong numbers (edge) is not actually the family the brief
asked for. Do not average these two rows into one "edge/roll coverage" headline
number for the deck -- they answer different questions and neither number
speaks for the other.

---

## 4. Sanity check: does it look sane on real images?

Four real GC10 test images, one per brief-relevant class, run through
`model.predict(imgsz=640, conf=0.15, device=mps)`:

**1. `crescent_gap`** (`img_01_425006200_01173...jpg`, 1 ground-truth box)
```
crescent_gap  conf=0.705  box=[478, 254, 767, 820]   <- overlaps gt box [474,218,782,821] closely
weld_line     conf=0.656  box=[680, 468, 2048, 556]  <- unlabelled 2nd detection, plausible, not checked against gt
```
Good match, high confidence.

**2. `crease`** (`img_01_425382900_00002...jpg`, 1 ground-truth box)
```
crease  conf=0.322  box=[944, 96, 2037, 315]
crease  conf=0.309  box=[939, 22, 1965, 296]
crease  conf=0.184  box=[941, 8, 1639, 291]
```
Right region, right class -- but three overlapping, low-confidence boxes where
one clean detection is warranted. This is exactly what a 0.292 AP50 predicts:
present, but imprecise and under-confident.

**3. `rolled_pit`** (`img_01_4402818600_00001...jpg`, 2 ground-truth boxes:
one `weld_line`, one `rolled_pit`)
```
weld_line  conf=0.668  box=[1632, 730, 2047, 812]   <- matches its gt box closely
(rolled_pit: NO DETECTION -- missed entirely)
```
The `weld_line` box is accurate. The `rolled_pit` box in the same frame (a small
box in the top-right corner) was **missed completely** at conf 0.15. This is not
a cherry-picked bad example -- it is one of the 4 out of 11 rolled_pit test
images the broader sweep in Section 3.1 found producing zero detections.

**4. `waist_fold`** (`img_03_4404374300_00045...jpg`, 1 ground-truth box)
```
waist_fold  conf=0.589  box=[631, 183, 1796, 930]   <- substantial overlap with gt [294,128,1609,951], smaller
```
Correct class, single clean box, real overlap, though it does not cover the
full labelled region -- consistent with `waist_fold` ground-truth boxes running
very large (Section 3.2).

**Sanity verdict: 3 of 4 hand-picked detections are recognisably correct and
land close to their ground truth; the 4th is a clean, representative miss on
the weakest class in the dataset (`rolled_pit`), which the numbers already
predicted and the 11-image sweep in Section 3.1 confirms is typical rather than
unlucky.**

---

## 5. What the brief asked for, what exists publicly, what was trained, and what would be needed to do it properly

**The brief:** ability to detect scratches, scale, roll marks, and edge cracks.

**What public data exists:** NEU-DET has `scratches` and `rolled-in_scale`
natively (already shipped). No public, openly-licensed dataset was found with an
"edge crack" class. GC10-DET (CC BY 4.0, via the Roboflow mirror of the original
Sensors 2020 release) is the closest public match for the remaining two
families: `rolled_pit` and `crease` for roll marks (direct), `crescent_gap` and
`waist_fold` for edge geometry (proxy).

**What was trained:** a 10-class YOLOv8n head (6 GC10 classes kept for
completeness, plus the 4 brief-relevant ones) on GC10-DET's full 2,294-image,
sequence-disjoint 70/15/15 split, at 640 px, 70 epochs, batch 16, seed 1337.
Standalone checkpoint at `models/yolov8n_gc10_640/weights/best.pt` -- **not**
merged into the shipped model or the joint model, and `resolve_weights()` is
untouched.

**What it scores:** mAP50 0.5821 overall on 333 held-out test images. Roll marks
(direct match): mean AP50 0.203 on 24 test instances (rolled_pit 13, crease 11)
-- weak, and honestly reported as such, with a live prediction sweep showing
4/11 rolled_pit images produce zero detections. Edge family (proxy, not
cracks): mean AP50 0.880 on 61 test instances (crescent_gap 40, waist_fold 21)
-- strong, but for edge-region geometry, not for cracks.

**What would be needed to do it properly:**

1. **A real edge-crack dataset**, labelled as such, or expert relabelling of a
   subset of strip frames by someone who can identify an actual crack versus an
   edge-geometry defect. Nothing public was found; this is a data-acquisition
   gap, not a modelling one.
2. **More `rolled_pit` and `crease` images.** 46 and 53 images respectively
   (across the whole 2,294-image dataset, not just the test split) is not enough
   to train a confident detector for either class. The model has learned
   something -- it is not at zero -- but the 4/11 miss rate on live images is a
   direct, measured consequence of how little data it saw.
3. **A larger input resolution or tiled training** for the small-box classes
   specifically, per `reports/gc10_dataset.md` section 7: whole-frame 640 px
   training shrinks `rolled_pit` boxes so 31% fall under 32 px on their short
   side, and `crease` at 43%. That table exists in the dataset report precisely
   so this training run's known ceiling was visible before the run, not
   discovered after.
4. **A merge decision**, analogous to the Gap 1 joint-model work: whether GC10
   coverage should stay a standalone checkpoint (current state), get folded into
   the joint model, or replace something -- deliberately out of scope for this
   report, which only evaluates what already exists.

**Bottom line for the deck:** this closes the roll-mark and edge-defect gap
*as far as public, honestly-labelled data allows, and no further.* Two of the
brief's four families now have measured, real coverage; two already had it. One
of the two new families is thin and weak; the other is strong but is a proxy,
not the thing named in the brief. All four of those qualifications belong in
the deck, not just in this report.

---

## Reproducing this report

```bash
# Dataset + model verification (Section 1)
.venv/bin/python -c "from ultralytics import YOLO; YOLO('models/yolov8n_gc10_640/weights/best.pt')"

# Held-out test evaluation (Section 2) -- iou=0.7 is the project's standing
# convention (ultralytics validator defaults), the same value every other val
# run in this repo uses
.venv/bin/yolo detect val \
    model=models/yolov8n_gc10_640/weights/best.pt \
    data=data/gc10-det/data.yaml \
    split=test imgsz=640 batch=16 device=mps conf=0.001 iou=0.7

# Sanity-check predictions (Section 4) -- see reports/gc10_coverage.json
# "sanity_check_predictions" and "rolled_pit_broad_sweep" for the exact
# image filenames and the predict() call used.
```

Raw numbers, full per-image sweep, and both families' combined statistics:
`reports/gc10_coverage.json`.
