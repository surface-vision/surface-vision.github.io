# GC10-DET: roll-mark and edge-defect coverage, and exactly how far it goes

**Built by** `src/prepare_gc10.py` (`.venv/bin/python src/prepare_gc10.py`)
**Output** `data/gc10-det/` (312 MB) + `data/gc10-det-raw/` (304 MB raw mirror)
**Source** `imaadd05/gc10-det` on HuggingFace, CC BY 4.0
**Generated** 2026-09-09

---

## 0. Read this before quoting anything below

The brief names four defect families: **scratches, scale, roll marks, edge
cracks**. Before this ingest the project covered the first two and had literally
zero coverage of the last two -- a repo-wide grep for "edge crack" returned no
hits outside one incidental line in `README.md`.

This dataset closes the gap **as far as public data allows, and no further**:

1. **There is no public "edge crack" class. There is no such class here either.**
   `crescent_gap` (3_yueyawan) and `waist_fold` (10_yaozhed) are edge-region
   *geometry* defects -- a scalloped bite out of the strip edge, and a folded-over
   edge. They are the closest honest proxies. They are not cracks. A model
   trained on them has never been shown an edge crack and must not be described
   as detecting one.
2. **The roll-mark class is thin.** `rolled_pit` is **85 boxes in 46 images**
   across the entire 2,294-image dataset. Any per-class AP computed on it will
   have a confidence interval wider than the number itself, and the same is true
   of `crease` (74 boxes in 53 images).

The defensible claim is: *we now have labelled data for the brief's roll-mark
family and a measured proxy for its edge family, and we state where each is a
proxy and where each is thin.* Anything stronger is not supported by this data.

---

## 1. Provenance and licence

| | |
|---|---|
| Mirror | https://huggingface.co/datasets/imaadd05/gc10-det |
| Licence | **CC BY 4.0** -- attribution required and given |
| Upstream of the mirror | Roboflow Universe (`nanjing-university-of-information-science-and-technology/gc10-det-zukta`), v5 COCO export |
| Original release | Lv, X.; Duan, F.; Jiang, J.J.; Fu, X.; Gan, L. *Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network.* Sensors 2020, 20(6), 1562 |
| Download measured | 2,299 files, **313.7 MB in 142 s** at 16 threads |

The original GC10-DET release is distributed over Baidu Pan with no stated
licence. The Roboflow mirror carries an explicit CC BY 4.0 grant, which is why it
is the copy used. Attribution is reproduced in `assets/README.md` and in
`data/gc10-det/manifest.json`.

---

## 2. Image geometry -- the project's first real strip frames

**Every one of the 2,294 images is 2048 x 1000, greyscale line-scan.** Verified,
not assumed: `manifest.json -> geometry` is `{"2048x1000": 2294}`, a single key.

This matters beyond the class list. NEU-DET is 200 x 200 defect-centred crops;
nothing in the repository before this was wider than 600 px, so
`DefectDetector.predict_tiled` -- shipped, tested, and named as a feature -- had
never had an image it could actually tile. A 2048 x 1000 frame is 3.2x the
detector's 256 px deployment input on the long side and forces the real tiling
path. Section 8 measures that.

---

## 3. How the split was built, and the leak that forced it

### 3.1 The upstream split leaks

The mirror ships its own 2,065 / 229 train/test split. It is not usable:

| | |
|---|---|
| sequence ids in upstream train | 449 |
| sequence ids in upstream test | 120 |
| **sequence ids present in both** | **102** |
| exact content-hash duplicates across the two | 0 |

GC10 filenames carry the original capture identity: `img_03_425005700_00156`
is camera 03, line-scan sequence `425005700`, frame 00156. Frames sharing a
sequence id are consecutive captures off the same running strip, and the camera
prefix does not separate them either -- `img_01` and `img_08` are two cameras
looking at the same strip position. So the 102 shared sequences are
near-duplicate content on both sides of the upstream split, even though zero
files are byte-identical. A model selected on that test set is being scored on
strip it has already seen.

**Content hashing alone would not have caught this** -- it reports 0 duplicates.
That is the entire reason the grouping key is the sequence id and not the hash.

### 3.2 What was built instead

All 2,294 images are pooled and re-cut 70/15/15 with **whole sequences kept
intact**, by iterative stratification: the class with the fewest unplaced boxes
picks its groups first (so `rolled_pit` at 85 boxes is served before `silk_spot`
at 884 crowds it out), and within a class the largest contributors are placed
first. Deterministic at `SEED = 1337` -- rebuilding from the same raw mirror
reproduces the split assignment digest `4e60d611c914f51b` exactly.

The ordering is not cosmetic. Placing groups in hash order instead gives a total
L1 ratio error of **1.217** across the ten classes (`silk_spot` lands 0.50 /
0.18 / 0.32 instead of 0.70 / 0.15 / 0.15); the contribution-ordered version
gives **0.058**. Both were measured before the choice was made.

### 3.3 Leakage check on what was written

Re-read from disk after writing, not asserted from memory:

| check | result |
|---|---|
| images hashed | 2,294 |
| unique SHA-256 content hashes | 2,294 |
| duplicate images *within* a split | 0 |
| **duplicate images *across* splits** | **0** |
| **line-scan sequences appearing in two splits** | **0** |

Both guards are enforced in `src/prepare_gc10.py` (`assert_disjoint`, which exits
non-zero on a cross-split duplicate) and independently re-checked in
`tests/test_gc10.py`
(`test_no_image_content_appears_in_two_splits`,
`test_no_line_scan_sequence_appears_in_two_splits`).

---

## 4. Counts per class per split

| # | class | pinyin | 汉字 | train | val | test | **total** | images |
|---|---|---|---|---|---|---|---|---|
| 0 | punching_hole | 1_chongkong | 冲孔 | 230 | 49 | 50 | 329 | 329 |
| 1 | weld_line | 2_hanfeng | 焊缝 | 359 | 77 | 77 | 513 | 512 |
| 2 | **crescent_gap** | 3_yueyawan | 月牙弯 | 185 | 40 | 40 | **265** | 264 |
| 3 | water_spot | 4_shuiban | 水斑 | 248 | 53 | 53 | 354 | 310 |
| 4 | oil_spot | 5_youban | 油斑 | 400 | 85 | 84 | 569 | 250 |
| 5 | silk_spot | 6_siban | 丝斑 | 626 | 129 | 129 | 884 | 734 |
| 6 | foreign_object | 7_yiwu | 异物 | 243 | 52 | 52 | 347 | 201 |
| 7 | **rolled_pit** | 8_yahen | 压痕 | 59 | 13 | 13 | **85** | 46 |
| 8 | **crease** | 9_zhehen | 折痕 | 52 | 11 | 11 | **74** | 53 |
| 9 | **waist_fold** | 10_yaozhed | 腰折 | 101 | 22 | 21 | **144** | 141 |
| | **boxes** | | | **2,503** | **531** | **530** | **3,564** | |
| | **images** | | | **1,641** | **320** | **333** | **2,294** | |
| | **sequences** | | | **252** | **108** | **107** | **467** | |

Two images carry no boxes at all and are kept as YOLO background frames.

Image counts land at 71.5 / 13.9 / 14.5 % rather than exactly 70/15/15, because
sequences are indivisible and the largest holds 202 frames. Box ratios come out
much tighter: the worst per-class deviation from 70/15/15 across all ten classes
and all three splits is **0.81 points** (`silk_spot` train, 70.8 %).

### Class-name mapping

The original labels are Chinese-pinyin with a leading ordinal. English names are
the class list written into `data.yaml` and are therefore the order the head
trains in; the index order follows the original 1..10 so a GC10 paper lines up
directly. The full mapping including a literal gloss is in
`data/gc10-det/manifest.json -> pinyin_mapping`.

One deliberate departure: **`7_yiwu` is named `foreign_object`, not
`inclusion`**, which is how most GC10 papers gloss it. NEU-DET already ships a
class called `inclusion`, and a later stage merges the two datasets into a single
16-class head, where a duplicated name would silently fuse two different defects.
异物 is literally "foreign matter", so the rename is also the better gloss.
`tests/test_gc10.py::test_no_gc10_class_collides_with_a_neu_det_class` pins the
16 names as distinct.

---

## 5. Mapping to the brief's four families

| brief family | GC10 classes | boxes (train/val/test) | status |
|---|---|---|---|
| **scratches** | -- | -- | Not in GC10. Covered by NEU-DET `scratches`. |
| **scale** | -- | -- | Not in GC10. Covered by NEU-DET `rolled-in_scale`. |
| **roll marks** | `rolled_pit`, `crease` | 111 / 24 / 24 (**159**) | Direct match, **thin** |
| **edge cracks** | `crescent_gap`, `waist_fold` | 286 / 62 / 61 (**409**) | **PROXY** -- edge geometry, not cracks |

Coverage after this ingest is therefore: two families with real in-domain data
(NEU-DET), one family with thin but direct data, one family with a stated proxy.
That is four out of four *addressed* and two out of four *covered well*. Saying it
that way is the point.

Out of the brief but retained: `punching_hole`, `weld_line`, `water_spot`,
`oil_spot`, `silk_spot`, `foreign_object` (2,796 boxes). `weld_line` is a
coil-join event a mill genuinely wants flagged. The rest are kept because
dropping a labelled box from an image that is being trained on would convert a
real defect into unlabelled background and teach the model to suppress it.

---

## 6. What is thin -- measured, not estimated

**`rolled_pit`, the direct roll-mark class, is 85 boxes in 46 images.** The
audit's figure of 64 was the upstream train split only; the pooled figure is 85,
and the image count is the number that actually bounds what can be learned. At
70/15/15 that leaves **13 boxes in val and 13 in test**. Consequences to state
before any training run, not after:

* Recall on that class moves by **7.7 points for every single box that flips**
  (1/13), and AP50 moves with it. The project's own bootstrap half-width on
  NEU-DET mAP50 with 446 test boxes is +/-0.025; on 13 boxes no honest interval
  will be narrower than the effect being claimed. **Report `rolled_pit` AP with
  its box count attached, or not at all.**
* `crease` (74 boxes, 53 images, 11 in val and test) is in the same position.
* `waist_fold` (144 boxes, 141 images) and `crescent_gap` (265 boxes, 264 images)
  are workable, and they are the proxy half of the edge family rather than the
  direct half.

A second honesty note on `waist_fold`: its median box covers **33.8 % of the
frame**. These are closer to whole-region labels than to localised defects, so a
high IoU-0.5 AP on `waist_fold` partly reflects the fact that a large box is easy
to overlap. `crescent_gap`, at a 5.3 % median area, is the more meaningful of the
two.

---

## 7. Box geometry, and the small-object risk for whichever stage trains this

Median box size per class, plus the fraction of boxes whose short side falls below
32 px once a 2048 px frame is letterboxed to a 640 px network input:

| class | median w x h (px) | median area (% of frame) | short side < 32 px at imgsz 640 |
|---|---|---|---|
| punching_hole | 108 x 60 | 0.31 | **100 %** |
| weld_line | 2014 x 74 | 6.09 | **94 %** |
| foreign_object | 67 x 69 | 0.24 | **83 %** |
| oil_spot | 114 x 102 | 0.57 | 62 % |
| crease | 628 x 118 | 3.73 | 43 % |
| rolled_pit | 211 x 129 | 1.50 | 31 % |
| water_spot | 257 x 322 | 4.17 | 9 % |
| crescent_gap | 267 x 408 | 5.29 | 6 % |
| silk_spot | 347 x 612 | 10.07 | 2 % |
| waist_fold | 935 x 814 | 33.83 | 1 % |

Whole-frame training at 640 px destroys the small classes: `punching_hole` boxes
become 34 x 19 px and `weld_line` becomes a 629 x 23 px sliver. The classes the
brief cares about survive better (`rolled_pit` 31 %, `crease` 43 %), but a
training stage that resizes 2048 -> 640 and reports a low `punching_hole` AP will
be measuring its own preprocessing. **Tiled training, or a much larger imgsz, is
the design question this table raises.** It is stated here rather than discovered
later; this ingest does not answer it and deliberately trains nothing.

Boxes per image: mean 1.55, median 1, max 11.

---

## 8. Two things this dataset gives the project beyond two class names

**Multi-defect frames, at a sample size that can carry a claim.** The brief asks
for *"ability to handle multiple defect types"*. NEU-DET's test split contains
12 multi-type frames, which the audit correctly refused to draw a conclusion
from. GC10 contains **476 multi-type images** (413 with two classes, 54 with
three, 9 with four), split 322 / 77 / 77. That is enough to measure.

**A working demonstration of tiled inference.** Measured just now, shipped
`yolov8n_neudet/best.pt` at imgsz 256, conf 0.15, on the three exported assets:

| asset | whole-frame detections | tiled detections | tiled latency |
|---|---|---|---|
| `strip_sample_edge_defect.jpg` | 0 | 20 | 299 ms* |
| `strip_sample_rollmark.jpg` | 2 | 17 | 106 ms |
| `strip_sample_weldline.jpg` | 1 | 7 | 92 ms |

\* first frame carries model warm-up. The latency is incidental here -- the point
is the detection counts.

The tiling code path works and the difference is dramatic -- which is the
demonstration that was impossible before, because no image in the repository was
wide enough to trigger it.

**Those detections are wrong, and that is the second finding.** The shipped model
knows only the six NEU-DET classes, so every box above is a cross-domain false
alarm on steel it has never seen. The class breakdown on the tiled runs is
`inclusion` 24, `pitted_surface` 6, `scratches` 5, `patches` 5,
`rolled-in_scale` 4 -- **`inclusion` is 55 % of them**, on a second dataset,
independently reproducing the audit's Severstal finding that `inclusion` is the
dominant out-of-domain hallucination and is exactly the class
`DispositionRules.hold_classes` treats as zero-tolerance. Max confidence reaches
0.773. This is not a measurement of GC10 accuracy -- no model has been trained on
GC10 -- it is the domain-shift failure showing up again on different steel.

---

## 9. Demo assets

Three frames copied into `assets/` (504 KB total), each 2048 x 1000, unmodified,
with CC BY 4.0 attribution in `assets/README.md`:

| file | shows | GC10 frame | split | labels |
|---|---|---|---|---|
| `strip_sample_rollmark.jpg` | four rolled pits on open strip | `img_03_4402117000_00003` | val | rolled_pit x4 |
| `strip_sample_edge_defect.jpg` | crescent bite out of the strip edge | `img_01_425006200_01173` | test | crescent_gap x1 |
| `strip_sample_weldline.jpg` | coil-join weld running 2,006 px across the 2,048 px strip | `img_07_3403337700_00899` | test | weld_line x1 |

Selected automatically by `export_assets`, ranked on: held-out split first (so
nothing shown in the demo is also a training image once a later stage trains on
this), then the headline class of the family, then a box that is neither a speck
nor the whole frame, then fewest other classes in shot, then frame brightness --
GC10 frames often catch the dark off-strip background, and a three-quarters-black
frame demos nothing.

`strip_sample_edge_defect.jpg` is captioned **edge defect, not edge crack**, in
the file's own README and in the manifest.

One measured property worth knowing before anyone builds a positional prior: the
strip edge is not at a fixed image column. `crescent_gap` boxes have a median
centre at 0.47 of the frame width and only 38 % sit in the outer quarter of the
image. The frames are crops in which the strip edge falls wherever the camera
put it, so image-relative x position is not a usable proxy for
distance-from-edge.

---

## 10. Reproducing this

```bash
.venv/bin/python src/prepare_gc10.py              # download + convert + split, ~3 min
.venv/bin/python src/prepare_gc10.py --skip-download   # re-split an existing mirror
.venv/bin/python -m pytest tests/test_gc10.py -v  # 50 tests
```

`tests/test_gc10.py` pins the class contract, the pinyin mapping, the
COCO->YOLO geometry (including clipping and degenerate-box rejection), the split
algorithm's determinism and ratios, both leakage guards, the manifest/disk
agreement, and the assets. It skips cleanly when `data/gc10-det` has not been
built. `tests/test_smoke.py` is untouched and still passes all 111 of its tests;
`pytest tests/test_smoke.py tests/test_gc10.py` is **161 passed**.

**Nothing here trains a model.** `models/yolov8n_neudet/weights/best.pt` and its
test mAP50 of 0.7524 are untouched.
