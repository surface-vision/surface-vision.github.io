# Severstal and joint NEU-DET + Severstal datasets

Built by `src/prepare_severstal.py` on 2026-09-09 15:48:47 with seed 1337.
Every number on this page is emitted by that script from the files it wrote.

## Why

NEU-DET has no defect-free image. The audit measured what that costs: the shipped checkpoint fires on ~91% of genuinely clean strip frames and the coil disposition chain returns HOLD. These two datasets put real defect-free steel, and real strip-geometry positives, into the training set.

## Source

- HuggingFace `Voxel51/severstal_steel_defects` (public, ungated), 18,074 images, every one 1600x256x3.
- Labelled `train` split only: 12568 frames = 6666 defective / 5902 verified defect-free. The dataset's own `test` split (5,506) carries `has_defect: null` and no mask -- it is the Kaggle unlabelled holdout and is never used here.
- Masks decode base64 -> zlib -> `.npy` to (256, 1600) uint8 where the pixel value is the class id 1-4. Boxes come from `scipy.ndimage.label` per class.
- Class names: the competition never published semantic names and the dataset card documents none, so classes are named after their mask value: `severstal_1` .. `severstal_4`.

## Image geometry and the tiling decision

NEU-DET frames are 200x200. Severstal frames are 1600x256, a 6.25:1 strip. Feeding a whole Severstal frame to YOLO at imgsz 256 letterboxes it to about 256x41 before padding, which squashes a 20 px defect to 3 px and puts the two datasets at wildly different pixels-per-mm in the vertical axis. So Severstal is **tiled**, not resized.

- Window 256x256, stride 224, 7 tiles per frame at x = 0, 224, 448, 672, 896, 1120, 1344; the last tile ends at exactly 1600, so coverage is complete with 32 px of overlap.
- Frames are exactly 256 px tall, so a 256 px square window needs no vertical tiling and **no resampling at all**: every crop is a byte-exact sub-rectangle of the source pixels and no box coordinate is ever approximated by a resize.
- 256 is this project's deployment `imgsz` and is within 28% of the NEU-DET frame size, so a NEU-DET defect and a Severstal defect reach the network at comparable magnification.

### Clipped defects: keep the box or drop the crop

For each connected component that lands in a crop, the box is recomputed as the tight bounding box of the mask pixels **inside the crop** (not by intersecting a full-frame box, so concave and diagonal defects stay exact). The fragment is kept as a labelled box if its clipped width and height are both >= 8 px and it holds >= 64 mask pixels. If any fragment fails that test the **entire crop is discarded** -- it is never emitted as a background image, because an unlabelled defect fragment teaches exactly the error this dataset exists to fix.

The threshold is absolute rather than a fraction of the parent component on purpose. Severstal defects are routinely 400-1000 px wide, so a 'retain >= 35% of the component' rule drops every tile of every wide defect; measured on 400 defective frames it lost the defect entirely on 18 frames versus 7 under this rule. Measured cost of the rule in this build: 242 crops discarded.

## What was subsampled, and how

Budget: `--max-positives 3000`, `--max-negatives 3000`, `--seed 1337`.

- **Positives.** 1228 defective source frames were consumed to reach 3001 positive crops (2.44 crops/frame). Frames are drawn **round-robin over their rarest present class**, not uniformly: the natural frame distribution is {1: 897, 2: 247, 3: 5150, 4: 801}, so a uniform draw of ~1200 frames would contain ~45 class-2 frames. Round-robin exhausts the rare classes first. This departs from the domain prior deliberately; the per-class instance counts below make the effect visible.
- **Negatives.** 1500 `has_defect=False` source frames, 2 randomly chosen tiles each, giving 3000 background crops. Two tiles per frame rather than all seven maximises source-frame diversity for a fixed crop budget.
- Negatives are taken **only** from frames the dataset certifies defect-free. Empty crops from defective frames are not used: Severstal's labels are known to be imperfect and 'unlabelled' is not 'clean'.

### Wall-clock arithmetic

NEU-DET's 1440 train images cost ~35 s/epoch at imgsz 320 on this machine (~24 ms/image). The joint train split is 6234 images, so ~152 s/epoch, i.e. roughly 71 epochs in a 3-hour budget before validation overhead. Total joint dataset: 7801 images.

## `data/severstal/` -- Severstal alone, 4 classes

| split | images | labelled | background | instances |
|---|---|---|---|---|
| train | 4794 | 2394 | 2400 | 3839 |
| val | 619 | 319 | 300 | 506 |
| test | 588 | 288 | 300 | 480 |
| **total** | **6001** | **3001** | **3000** | **4825** |

| class | idx | train | val | test | total |
|---|---|---|---|---|---|
| severstal_1 | 0 | 1055 | 117 | 141 | 1313 |
| severstal_2 | 1 | 321 | 27 | 31 | 379 |
| severstal_3 | 2 | 1415 | 206 | 197 | 1818 |
| severstal_4 | 3 | 1048 | 156 | 111 | 1315 |

## `data/joint/` -- NEU-DET 0-5 + Severstal 6-9, one index space

NEU-DET indices 0-5 and its train/val/test membership are **unchanged**, so the shipped `yolov8n_neudet` checkpoint warm-starts on the first six classes and the held-out NEU-DET test numbers stay directly comparable.

| split | images | labelled | background | instances |
|---|---|---|---|---|
| train | 6234 | 3834 | 2400 | 7174 |
| val | 799 | 499 | 300 | 914 |
| test | 768 | 468 | 300 | 926 |
| **total** | **7801** | **4801** | **3000** | **9014** |

| class | idx | train | val | test | total |
|---|---|---|---|---|---|
| crazing | 0 | 538 | 72 | 79 | 689 |
| inclusion | 1 | 819 | 103 | 89 | 1011 |
| patches | 2 | 702 | 80 | 99 | 881 |
| pitted_surface | 3 | 339 | 47 | 46 | 432 |
| rolled-in_scale | 4 | 504 | 55 | 69 | 628 |
| scratches | 5 | 433 | 51 | 64 | 548 |
| severstal_1 | 6 | 1055 | 117 | 141 | 1313 |
| severstal_2 | 7 | 321 | 27 | 31 | 379 |
| severstal_3 | 8 | 1415 | 206 | 197 | 1818 |
| severstal_4 | 9 | 1048 | 156 | 111 | 1315 |

## Leakage check (script output, verbatim)

Splits are cut at the **source-frame** level, so all seven tiles of a Severstal frame land in the same split. Two independent checks then run over the written files: source-frame id disjointness, and sha256 of the decoded pixels of every image on disk.

```
[leakage] severstal source-frame disjointness (crops of one frame never straddle a split)
           train frames=2182 vs val frames=273: 0 shared source ids
           train frames=2182 vs test frames=273: 0 shared source ids
           val frames=273 vs test frames=273: 0 shared source ids
           VERDICT: PASS - no source frame straddles a split
[leakage] data/severstal: content-hash split check
           train n=4794 vs val n=619: 0 shared image hashes
           train n=4794 vs test n=588: 0 shared image hashes
           val n=619 vs test n=588: 0 shared image hashes
           intra-split duplicate hashes: {'train': 0, 'val': 0, 'test': 0}
           VERDICT: PASS - all splits disjoint by content hash
[leakage] data/joint: content-hash split check
           train n=6234 vs val n=799: 0 shared image hashes
           train n=6234 vs test n=768: 0 shared image hashes
           val n=799 vs test n=768: 0 shared image hashes
           intra-split duplicate hashes: {'train': 1, 'val': 0, 'test': 0}
           VERDICT: PASS - all splits disjoint by content hash
[leakage] data/neu-det: content-hash split check
           train n=1440 vs val n=180: 0 shared image hashes
           train n=1440 vs test n=180: 0 shared image hashes
           val n=180 vs test n=180: 0 shared image hashes
           intra-split duplicate hashes: {'train': 1, 'val': 0, 'test': 0}
           VERDICT: PASS - all splits disjoint by content hash
```

## Negative verification

`has_defect=False` samples carry no `ground_truth` object at all, i.e. zero annotated defect pixels by construction. The builder still materialises the all-zero (256, 1600) mask and asserts the crop's slice sums to zero and contains none of the values 1-4. **3000 / 3000 background crops passed that check; 0 failed.**

## Independent re-derivation of every label (script output, verbatim)

After writing, the builder re-opens `samples.json`, re-decodes each source mask, re-runs connected components and the clipping rule, and compares the result against the `.txt` files on disk -- in both index spaces. Nothing from the build is reused, so a bug in the crop path cannot hide behind itself.

```
[verify] independent re-derivation from samples.json masks and source frames
           3001 labelled crops (4825 boxes) + 3000 background crops from 2728 source frames re-derived
           label mismatches: 0
[verify] 600 sampled crops vs their source-frame rectangle: mean abs pixel delta 0.251/255, worst 8/255 (JPEG q95 re-encode only; the crop rectangle itself is exact)
           wrong-shape crops: 0; joint copies differing from severstal copies: 0
           VERDICT: PASS - every label re-derives from the source mask
```

## Ultralytics loader check (script output, verbatim)

```
[loader] severstal: nc=4 names=['severstal_1', 'severstal_2', 'severstal_3', 'severstal_4']
           train: 4794 images, 2400 backgrounds, 3839 instances, classes present [0, 1, 2, 3]
           val: 619 images, 300 backgrounds, 506 instances, classes present [0, 1, 2, 3]
           test: 588 images, 300 backgrounds, 480 instances, classes present [0, 1, 2, 3]
[loader] joint: nc=10 names=['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches', 'severstal_1', 'severstal_2', 'severstal_3', 'severstal_4']
           train: 6234 images, 2400 backgrounds, 7172 instances, classes present [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
           val: 799 images, 300 backgrounds, 913 instances, classes present [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
           test: 768 images, 300 backgrounds, 926 instances, classes present [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Known quirks, inherited not introduced

- `data/neu-det/train` contains one byte-identical image pair (`patches_101.jpg` == `patches_105.jpg`). Both are in **train**, so it is not split leakage, and the joint set carries it forward unchanged rather than silently editing NEU-DET.
- Ultralytics reports 3 duplicate label rows while scanning the joint set (`crazing_120`, `inclusion_62`, `patches_198`) and drops them, which is why its instance totals read 7172/913 against the 7174/914 counted from the raw `.txt` files. All three are pre-existing NEU-DET annotations, copied verbatim. Severstal contributes zero duplicates.
- Crops are re-encoded as JPEG q95. The crop *rectangle* is exact -- no resampling -- but the re-encode costs a mean absolute pixel delta of 0.2513/255 against the source frame (worst 8/255).

## Build provenance

- frames read from cache: 2816
- frames reused from prior audit scratch dirs: 0
- frames downloaded from the hub this run: 0
- crops rejected as byte-identical duplicates: 1
- fetch/decode failures: 0
- machine-readable manifest: `data/severstal/manifest.json`, `data/joint/manifest.json`

