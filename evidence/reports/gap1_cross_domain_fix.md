# Gap 1: the model had never seen clean steel

**Status:** fixed and measured. **Recommendation:** ship `models/yolov8n_joint/weights/best.pt`
**alongside** the current default, do not repoint `resolve_weights` in this task.
The joint model wins on every axis -- clean-frame false alarms 93.7% -> 32.5%,
cross-domain ROC AUC 0.608 -> 0.958, Severstal defect recall 72.0% -> 86.6%, and
NEU-DET held-out mAP50 *up* 0.7524 -> 0.7642 -- but `src/inference.py` raises
`KeyError` on a 10-class head and nobody has set a severity tier for the four
Severstal classes, so making it the default is a separate, named piece of work
(section 10).

Every number below was produced by a command printed next to it, on this machine,
on 2026-09-09. Nothing is copied from a paper and nothing is estimated.

---

## 1. The problem, as measured

`models/yolov8n_neudet/weights/best.pt` was fine-tuned on NEU-DET. Every one of
NEU-DET's 1,800 images contains a defect, so the optimisation never once had to
answer the question a mill actually asks -- *is this frame clean?* It was never
penalised for saying "defect" to a piece of good steel, because it was never
shown a piece of good steel.

`src/cross_domain.py` measures what that costs on 505 held-out, certified
defect-free Severstal strip frames (4,040 200x200 tiles) and 191 mask-labelled
defective frames:

| shipped model, conf 0.15 | value | 95% CI (image-clustered bootstrap) |
|---|---|---|
| clean **frame** false alarm | 93.7% | [91.5%, 95.6%] |
| clean **crop** false alarm | 55.1% | [52.2%, 58.0%] |
| boxes per clean crop | 2.02 | |
| defect crop recall | 72.0% | [67.3%, 76.4%] |
| ROC AUC, clean vs defective | **0.608** | [0.576, 0.640] |
| ROC AUC, same model in domain | 0.968 | [0.948, 0.984] |

A verified-clean coil put through the project's own disposition chain
(`src/report.py`, rules untouched) comes out **HOLD** on four separate triggers,
the loudest being a 55.2% defect rate and 101 frames of "inclusion" -- a
zero-tolerance class -- on steel that has no defect in it at all.

Deck slide 5 claims the backbone features transfer unchanged. AUC 0.608 against
0.968 is the evidence against that sentence. 0.5 is a coin.

## 2. The trap, which the auditor already fell into on purpose

The obvious repair is to warm-start on NEU-DET plus Severstal **negatives**. The
auditor measured it: clean-frame false alarms collapse from 94% to 5.8% at
essentially zero in-domain cost (mAP50 -0.0016, mAP50-95 +0.0073). It is
worthless. Severstal defect recall collapses to **0.057** and the AUC does not
move. The model has not learned what a defect is; it has learned that this
texture is not NEU-DET, so it must be clean. Importing negatives from a domain
that contributes no positives can only ever train a *domain* classifier.

The fix has to put Severstal positives and Severstal negatives into the *same*
optimisation, so that "clean" and "defective" have to be separated *within* the
new domain rather than between the two domains.

## 3. What I checked before training on the dataset

`data/joint` was built by another agent. I did not take its manifest on trust;
`src/train_joint.py verify` re-derives every claim from the files themselves.

```
.venv/bin/python -m src.train_joint verify \
  --samples-json <audit scratch>/samples.json \
  --json reports/gap1_dataset_verification.json
```

```
[PASS] class vocabulary: nc=10 names=['crazing', 'inclusion', 'patches', 'pitted_surface',
       'rolled-in_scale', 'scratches', 'severstal_1', 'severstal_2', 'severstal_3', 'severstal_4']
[PASS] NEU-DET indices 0-5 unchanged
[PASS] counts recounted, train: 6234 images (3834 labelled / 2400 background), 7174 instances -- manifest agrees: True
[PASS] counts recounted, val:    799 images ( 499 labelled /  300 background),  914 instances -- manifest agrees: True
[PASS] counts recounted, test:   768 images ( 468 labelled /  300 background),  926 instances -- manifest agrees: True
[PASS] label geometry in range: every box has 0 < w,h <= 1 and lies inside the tile
[PASS] class ids stay in their own domain: NEU-DET images carry only classes 0-5, Severstal crops only 6-9
[PASS] NEU-DET images/labels byte-identical and in the same split: {'train': 1440, 'val': 180, 'test': 180}
       images compared; 0 mismatches, 0 split differences
[PASS] splits are content-hash disjoint: shared image hashes {'train^val': 0, 'train^test': 0, 'val^test': 0}
[PASS] no Severstal source frame straddles a split: shared source frames {'train^val': 0, 'train^test': 0, 'val^test': 0}
[PASS] background crops come only from has_defect=False source frames: 0 background crops from
       defective frames, 0 labelled crops from clean frames, 0 frames not in samples.json
[PASS] crops are the source frame's own pixels: 120 random crops re-cut from data/.severstal_cache;
       worst absolute channel delta 6 (JPEG re-encode only, no resampling); 0 sources unavailable
[FAIL] data/joint respects the cross_domain holdout rule: 2770 train+val crops come from frames
       cross_domain.py scores (seed 'jsw-ps1-cross-domain-holdout-v1', frac 0.5)
```

Nothing here trusts the manifest. Counts are recounted from the files on disk, the
1,800 NEU-DET images are compared byte-for-byte (SHA-256) against `data/neu-det`
and confirmed to sit in the same split they always did, background labels are
re-read, split disjointness is re-hashed, background crops are cross-checked
against Severstal's own `has_defect` flag in `samples.json`, and 120 randomly
chosen Severstal crops are re-cut from the cached source frames to confirm they
are the frame's own pixels (worst channel delta 6/255, which is JPEG re-encode,
not resampling -- the tiler does not resize).

Everything that matters passed. Two remarks on the two lines that are not a
plain pass:

* **Label geometry.** 514 of 9,014 boxes overflow the tile edge by up to
  5e-7 in normalised units -- 1.3e-4 of a pixel on a 256 px tile. That is the
  6-decimal-place rounding in the label writer, not a bad box, so the check
  carries an explicit one-ulp tolerance rather than pretending the boxes are
  exact.
* **The holdout rule.** This one is a real finding and section 4 is about it.

## 4. The leak I found, and why the training set is smaller than the one I was handed

`src/cross_domain.py` splits every Severstal source frame with a deterministic
hash, `is_holdout(stem, 0.5)`, and scores only the holdout half. Its own
docstring reserves the other half: *"the other half is what a fix is allowed to
train on; nothing in this file ever scores a frame the split assigns to train, so
a post-fix record stays honest without anyone having to remember."*

`src/prepare_severstal.py` does not know that rule exists. Measured:

| split | Severstal source frames | on the harness's **scored** side | crops affected |
|---|---|---|---|
| train | 2,182 | 1,133 | 2,465 |
| val | 273 | 133 | 305 |
| test | 273 | 130 | 276 |

Of those, **109 clean and 24 masked-defective train-split frames are frames the
harness actually scores today**. Training on them would have inflated every
cross-domain number in section 7 by construction, and a judge re-running
`cross_domain.py` against a larger pull of the same public dataset would have
silently contaminated the rest.

So nothing here trains on `data/joint`. `src/train_joint.py build` derives
`data/joint_xdsafe` -- symlinks only, no pixels copied, `data/joint` never
written to -- dropping from **train and val** every Severstal crop whose source
frame satisfies `is_holdout(stem, 0.5)`. The rule is the harness's own and does
not reference which frames the auditor happened to download, so the record stays
honest if the harness's data root grows. `test` is left whole: nothing trains on
it under either rule.

```
.venv/bin/python -m src.train_joint build
```

| split | images in | kept | dropped | NEU kept | Severstal kept | labelled | background |
|---|---|---|---|---|---|---|---|
| train | 6,234 | **3,769** | 2,465 | 1,440 | 2,329 | 2,579 | 1,190 |
| val | 799 | **494** | 305 | 180 | 314 | 334 | 160 |
| test | 768 | 768 | 0 | 180 | 588 | 468 | 300 |

The cost is real and I am not hiding it: the rule throws away **51% of the
Severstal training crops** (4,794 -> 2,329). I paid it because an unimpeachable
cross-domain number is the entire point of this deliverable, and because 1,190
certified-clean training crops is already an order of magnitude more clean steel
than the shipped model has ever seen (zero).

## 5. Warm start: the head is transplanted, not thrown away

The shipped checkpoint has a 6-class head; the joint dataset needs 10.
Ultralytics' default transfer (`intersect_dicts`) drops every tensor whose shape
changed, which discards the three learned 1x1 classification convolutions and
re-randomises all ten classes -- a needless loss, since only four output channels
are actually new.

`src/train_joint.py warmstart` builds the 10-class model, runs the standard
transfer for backbone and neck (349 of 355 tensors, 98.3%), then copies the
learned `cv3[i][-1]` weight and bias into output channels 0-5 and leaves 6-9 at
Ultralytics' own `bias_init` prior. Box regression (`cv2`) is class-independent
and transfers unchanged. Net effect: **100% of the shipped model's still-meaningful
parameters survive.**

```
.venv/bin/python -m src.train_joint warmstart
```

The check that this is real, rather than a claim: the widened, untrained
checkpoint reproduces the shipped model's NEU-DET held-out test metrics to eight
significant figures.

| checkpoint | data yaml | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| `yolov8n_neudet/best.pt` (6 classes) | `data/neu-det/data.yaml` | 0.7524475543783158 | 0.3967094751925778 | 0.6960037 | 0.6867121 |
| `warmstart_10cls.pt` (10 classes) | `data/joint_xdsafe/eval_neu_test.yaml` | 0.7524475543783158 | 0.3967094751925778 | 0.6960038 | 0.6867122 |

That single row does double duty: it proves the transplant, and it proves that
scoring a 10-class checkpoint through `eval_neu_test.yaml` is metric-identical to
scoring a 6-class one through the canonical NEU-DET yaml. Every in-domain
comparison below rests on it.

## 6. What I trained

```
.venv/bin/python -m src.train_joint train --epochs 200 --patience 40 \
    --imgsz 320 --batch 32 --name yolov8n_joint
```

Identical to `src/train_detector.py` in every respect that is not the point of
the experiment: imgsz 320, batch 32, `optimizer="auto"` (SGD, lr0 0.01, cosine),
seed 1337, and the same augmentation recipe (`fliplr/flipud 0.5, degrees 10,
scale 0.4, translate 0.1, shear 2, mosaic 1.0, close_mosaic 15, mixup 0.1,
hsv_h 0, hsv_s 0.3, hsv_v 0.4`). A test asserts that recipe is byte-for-byte the
one in `train_detector.py`, so "same recipe" is checked, not claimed. The only
changes are the data and the warm start.

**Epoch budget, from measurement not guesswork.** A 2-epoch probe on the real
dataset took 36.1 s and 34.7 s per epoch (118 iterations at batch 32, plus
validation on 494 images). Against a ~3 h budget that allows about 290 epochs; I
chose **200 with patience 40**, projected 2 h 05 m, leaving headroom inside the
budget for the four evaluation passes. 200 epochs x 118 iterations is 23,600
optimisation steps against the shipped run's 6,750, so the budget is not the
binding constraint -- convergence is, which is what patience is for.

**What actually happened.** 200/200 epochs, no early stop -- validation fitness
was still improving at epoch 179, which is where `best.pt` comes from
(fitness 0.4254, joint-val mAP50 0.6993, mAP50-95 0.3950). Median epoch 44.6 s.

Two of the 200 epochs took 943 s and 1,196 s against a 44.6 s median. That is not
the model: `time.perf_counter()` (which does not advance while macOS is asleep)
measured 9,266 s of compute, while Ultralytics' `time.time()`-based clock recorded
11,430 s of wall time. The 2,164 s difference is almost exactly the 2,139 s those
two epochs absorbed, so the machine suspended for ~36 minutes mid-run. **Compute:
2 h 34 m. Wall clock: 3 h 10 m.** The budget was met on the resource that was
actually being spent.

Artefacts: `models/yolov8n_joint/weights/best.pt` (SHA-256 `198e9b4bec5b4809...`),
`results.csv`, `args.yaml`, `train_console.log`, `train_result.json`.

---

# The three evaluation axes

## 7a. In domain: NEU-DET held-out test

```
.venv/bin/python -m src.train_joint eval \
    --weights models/yolov8n_joint/weights/best.pt \
    --json reports/gap1_detection_metrics.json
.venv/bin/python -m src.train_joint eval \
    --weights models/yolov8n_neudet/weights/best.pt \
    --json reports/gap1_detection_metrics_baseline.json
```

Same 180 images, same 446 instances, same labels. The shipped 6-class checkpoint
goes through `data/neu-det/data.yaml`; the 10-class one through
`data/joint_xdsafe/eval_neu_test.yaml`. Section 5 shows those two paths are
metric-identical.

| model | imgsz | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| shipped `yolov8n_neudet` | 256 | 0.7524 | 0.3967 | 0.6960 | 0.6867 |
| shipped `yolov8n_neudet` | 320 | 0.7286 | 0.3926 | 0.7322 | 0.6018 |
| **joint `yolov8n_joint`** | **256** | **0.7642** | **0.4008** | 0.6946 | 0.7055 |
| **joint `yolov8n_joint`** | 320 | 0.7454 | 0.3956 | 0.6949 | 0.6818 |

**The in-domain number did not drop. It went up**: +0.0118 mAP50 and +0.0041
mAP50-95 at the deployment size of 256, and +0.0168 mAP50 at 320. I want to be
careful about how much weight that carries -- it is one run against one run on 180
images, with no paired bootstrap behind it, so the honest reading is **"no
measurable in-domain cost"**, not "the joint model is better in domain". What it
does rule out is the outcome the task was worried about: there is no trade-off to
present here, because nothing was traded.

Per-class AP50 on NEU-DET test at imgsz 256:

| class | shipped | joint | delta |
|---|---|---|---|
| crazing | 0.4443 | 0.5854 | +0.1411 |
| inclusion | 0.8272 | 0.7869 | -0.0403 |
| patches | 0.9486 | 0.9243 | -0.0243 |
| pitted_surface | 0.7558 | 0.8142 | +0.0584 |
| rolled-in_scale | 0.6295 | 0.6031 | -0.0264 |
| scratches | 0.9093 | 0.8711 | -0.0382 |

Four classes lose 2-4 points; the two weakest classes gain 14 and 6. That pattern
is what you would expect from 1,190 clean-steel negatives: the model stops
carpeting low-contrast texture with boxes, which helps `crazing` (a texture class
that was being confused with everything) and costs a little recall on the strong
classes.

## 7b. Cross domain, supervised: Severstal held-out test

588 held-out Severstal crops, 480 instances, from source frames disjoint at frame
level from anything trained on. Same command as 7a.

| imgsz | Severstal mAP50 | Severstal mAP50-95 | mean recall |
|---|---|---|---|
| 256 | 0.4559 | 0.2203 | 0.4467 |
| **320** | **0.5656** | **0.2634** | **0.5477** |

| Severstal class | AP50 @256 | AP50-95 @256 | P | R | test instances |
|---|---|---|---|---|---|
| severstal_1 | 0.4943 | 0.1951 | 0.5746 | 0.4610 | 141 |
| severstal_2 | 0.4782 | 0.2501 | 0.5230 | 0.5661 | 31 |
| severstal_3 | 0.3317 | 0.1370 | 0.5166 | 0.3092 | 197 |
| severstal_4 | 0.5193 | 0.2990 | 0.5565 | 0.4505 | 111 |

The shipped model's Severstal mAP50 is **0 by construction** -- it has no output
channel that can emit a Severstal class, so every Severstal instance is an
unmatched ground-truth box. Quoting that as a comparison would be theatre; the
meaningful cross-model comparison is the class-agnostic recall in 7c, where both
models are asked only "is there a defect here".

Note the model prefers imgsz 320 on Severstal (+0.11 mAP50) and 256 on NEU-DET.
That is a field-of-view effect, not a contradiction: Severstal crops are 256 px of
a 1600 px strip and its defects are large, so more input resolution helps; NEU-DET
frames are 200 px and already sit at 1.28x at imgsz 256.

## 7c. Cross domain, the actual gap: `src/cross_domain.py`

```
.venv/bin/python -m src.train_joint crossdomain -- \
    --weights models/yolov8n_joint/weights/best.pt --tag joint \
    --json reports/gap1_cross_domain.json \
    --md reports/gap1_cross_domain.md \
    --plot reports/gap1_cross_domain_fa_vs_recall.png
```

Identical protocol to the baseline record: same 505 held-out clean frames (4,040
200x200 tiles), same 191 masked defective frames (546 labelled defective tiles),
same imgsz 256, same iou 0.45, same 4,000-draw image-clustered bootstrap. The
baseline row is the record the harness already held; it was not re-run and not
re-derived.

That the two rows score the *same* population is not a claim, it is a hash: both
records carry `clean_manifest_sha256 1688fe2263c1...` and
`defect_manifest_sha256 5011fd23e454...`. `src/cross_domain.py` was last modified
at 16:04 and the baseline record was written at 16:10, so both rows also come
from the same code.

And the training set is disjoint from that population by construction, verified
rather than assumed: of the 696 frames `cross_domain.py` actually scores,
**0** appear among the 1,049 Severstal source frames in `joint_xdsafe/train` or
the 140 in `joint_xdsafe/val`. A test (`test_holdout_safe_tree_trains_on_nothing_
the_harness_scores`) asserts the stronger hash-level property on every image in
the tree.

| conf | clean-FRAME FA base | clean-FRAME FA joint | clean-crop FA base | clean-crop FA joint | defect crop recall base | defect crop recall joint |
|---|---|---|---|---|---|---|
| 0.05 | 98.0% | **52.9%** | 68.2% | **11.9%** | 87.7% | **93.0%** |
| 0.10 | 96.0% | **40.2%** | 60.4% | **7.8%** | 80.6% | **89.9%** |
| **0.15** | **93.7%** | **32.5%** | **55.1%** | **5.9%** | **72.0%** | **86.6%** |
| 0.20 | 91.1% | **26.3%** | 50.1% | **4.6%** | 63.9% | **83.2%** |
| 0.25 | 86.7% | **21.8%** | 44.4% | **3.6%** | 57.0% | **80.0%** |
| 0.30 | 85.1% | **18.4%** | 38.2% | **3.1%** | 51.8% | **75.5%** |
| 0.40 | 77.4% | **13.1%** | 29.4% | **2.1%** | 41.8% | **69.8%** |
| 0.50 | 71.7% | **7.7%** | 23.5% | **1.3%** | 30.2% | **62.1%** |
| 0.60 | 57.8% | **4.8%** | 16.3% | **0.7%** | 18.3% | **50.5%** |
| 0.70 | 39.8% | **4.0%** | 8.9% | **0.6%** | 8.2% | **36.6%** |
| 0.80 | 17.4% | **2.2%** | 3.0% | **0.3%** | 2.7% | **17.2%** |

At the shipped operating point, clean-frame false alarms fall **93.7% -> 32.5%**
and clean-crop false alarms **55.1% -> 5.9%** (a 9.3x reduction), while defect
recall **rises 72.0% -> 86.6%**. The joint model is strictly better on both axes
at every threshold in the sweep -- there is no point on the baseline's curve that
the new curve does not dominate.

**Separation (ROC AUC, 4,000-draw bootstrap resampling whole source frames):**

| model | cross-domain AUC | 95% CI | in-domain AUC | 95% CI |
|---|---|---|---|---|
| shipped | 0.6080 | [0.5761, 0.6398] | 0.9679 | [0.9476, 0.9841] |
| **joint** | **0.9584** | **[0.9478, 0.9682]** | 0.9398 | [0.9053, 0.9698] |

Cross-domain separation goes from **near chance to 0.958**, with a confidence
interval that does not come within 0.30 of the baseline's. In-domain separation
falls 0.968 -> 0.940; the intervals overlap substantially ([0.948, 0.984] vs
[0.905, 0.970]), so that difference is not established, but it is not zero either
and I am not going to claim it is.

**This is the result that distinguishes the fix from the trap.** Negatives alone
gave the auditor a 5.8% false alarm rate with Severstal recall 0.057 and an
unmoved AUC -- a domain classifier. Joint training gives 5.9% clean-crop false
alarms with Severstal recall 0.866 and AUC 0.958 -- a defect classifier. Almost
the same false alarm number; the opposite model.

![false alarm vs recall](gap1_cross_domain_fa_vs_recall.png)

---

## 8. The disposition chain on a verified-clean coil

This is the sentence the mill cares about, so it gets its own protocol. 60
certified defect-free Severstal strip frames, cut into **the identical 480
200x200 tiles for both models** (verified: the two tile directories have the same
file listing), through `src/report.py` with `DispositionRules` untouched.

```
# tiles are written by the 7c run into reports/cross_domain_coil/joint/
.venv/bin/python -m src.report --images reports/cross_domain_coil/joint \
    --weights models/yolov8n_neudet/weights/best.pt --conf 0.15 --imgsz 256 --no-cam ...
.venv/bin/python -m src.train_joint coil --tier medium -- \
    --images reports/cross_domain_coil/joint \
    --weights models/yolov8n_joint/weights/best.pt --conf 0.15 --imgsz 256 --no-cam ...
```

| model | conf | verdict | defect rate | detections | p95 severity | triggers |
|---|---|---|---|---|---|---|
| shipped | 0.15 | **HOLD** | 55.2% (265/480) | 960 | 89.0 | defect rate, p95 severity, 62 critical-band frames, **101 frames of "inclusion"** (zero-tolerance) |
| shipped | 0.40 | **HOLD** | 26.5% (128/480) | 276 | 83.0 | defect rate, p95 severity, 30 critical-band frames, 55 frames of "inclusion" |
| **joint** | **0.15** | **DOWNGRADE** | **4.8% (23/480)** | **25** | **0.0** | defect rate above the 2% accept limit; 4 frames in the high band |
| **joint** | 0.25 | DOWNGRADE | 2.5% (12/480) | 12 | 0.0 | defect rate above the 2% accept limit; 2 frames in the high band |
| **joint** | 0.40 | DOWNGRADE | **1.0% (5/480)** | 5 | 0.0 | 1 frame in the high band (defect rate now *inside* the accept limit) |

**The clean coil is no longer held.** Every hard HOLD trigger is gone at the
shipped operating point: the defect rate falls from 55.2% to 4.8%, the 101
zero-tolerance "inclusion" frames fall to zero, the 62 critical-band frames fall
to zero, and p95 severity falls from 89.0 to 0.0 (fewer than 5% of tiles carry
any box at all, so the 95th percentile of the severity distribution is literally
zero). 960 detections become 25.

Per-class frame counts on the 480 clean tiles at conf 0.15, joint model:
`inclusion 0` (was 101), `crazing 0` (was 5), `rolled-in_scale 0` (was 15),
`patches 1` (was 102), `pitted_surface 2` (was 76), `scratches 2` (was 65), plus
`severstal_1 4, severstal_2 3, severstal_3 4, severstal_4 8`. Severity bands:
457 clean, 9 low, 10 medium, 4 high, **0 critical** (was 62). Five NEU-class
false positives on clean steel, down from 265 flagged frames.

**It is not yet a full ACCEPT, and here is exactly what stands in the way.** At
conf 0.40 the coil clears the 2% defect-rate limit (1.0%) and is blocked by a
single frame in the "high" severity band. All five surviving detections are:

```
1607a314f_x00800_y00028  medium  38.7  severstal_4 conf 0.584 area 0.074
1607a314f_x01000_y00028  medium  43.2  severstal_4 conf 0.665 area 0.107
382dc8f4b_x00800_y00028  medium  45.2  patches     conf 0.821 area 0.082
39b187684_x00400_y00028  medium  34.0  severstal_1 conf 0.541 area 0.023
3b2b465b0_x01400_y00028  high    56.3  severstal_4 conf 0.729 area 0.394
```

Four of the five are Severstal classes, whose severity tier **nobody knows**. The
Severstal competition never published semantic names for its four defect ids, so
`DEFECT_INFO` has no honest entry for them and I refuse to invent one. The
verdict is therefore reported as a function of that unknown:

| placeholder tier for `severstal_*` | verdict @ conf 0.15 | verdict @ conf 0.40 |
|---|---|---|
| low | DOWNGRADE | **ACCEPT** |
| medium (default used above) | DOWNGRADE | DOWNGRADE |
| high | HOLD | HOLD |
| critical | HOLD | HOLD |

So the precise, defensible claim is: **a verified-clean coil that the shipped
model holds is, with the joint model, downgraded at the shipped threshold and
accepted at conf 0.40 under a low-severity reading of the unknown classes.** The
class-independent number -- the defect rate, which no severity assumption can
move -- goes 55.2% -> 4.8% -> 1.0%. Setting those four tiers is a quality-department
decision, and it is listed as a ship prerequisite in section 10.

**Whole-frame geometry.** `cross_domain.py` also runs the naive path: each
1600x256 strip handed to the detector whole. Baseline HOLD (defect rate 50.0%);
joint HOLD (defect rate 46.7%, 32 detections vs the baseline's 53, p95 severity
50.6 vs 72.0, and no zero-tolerance or critical-band trigger). Letterboxing
1600x256 to imgsz 256 squashes the strip to roughly 256x41 -- a 20 px defect
becomes 3 px -- and the joint model was trained on 256 px tiles, not on squashed
strips, so this geometry is out of distribution for it too. It improves but does
not pass, and it should not be used; the tiled path is the deployed one.

## 9. Re-picking the operating threshold, on VALIDATION only

```
.venv/bin/python -m src.train_joint threshold \
    --weights models/yolov8n_joint/weights/best.pt \
    --json reports/gap1_operating_point.json
```

Population: `data/joint_xdsafe/val` -- 494 frames, of which **334 labelled** (180
NEU-DET + 154 Severstal) and **160 certified defect-free Severstal crops**. Test
data is not touched. Every frame is scored once at conf 0.01 and the whole sweep
is filtering of that cache, which is exact rather than approximate: NMS is greedy
in descending confidence, so a box above `t` can only have been suppressed by a
box above `t`.

The cost model is `false_alarm.COST_MODEL` unchanged -- miss:false-alarm 12:1,
prevalence 0.05, detection floor 0.90, so effective lambda 0.6316 -- and the
selection rule is `false_alarm.choose_threshold`. One thing *is* better than the
shipped study: `F(t)` is measured on genuinely defect-free steel, so it needs
none of the Poisson area extrapolation that `false_alarm.py` had to test and
reject.

| conf | D(t) localised, all | D(t) NEU | D(t) Severstal | flag rate, all | F(t) clean FA | boxes/clean crop |
|---|---|---|---|---|---|---|
| 0.01 | 0.9042 | 0.9556 | 0.8442 | 0.9850 | 0.2812 | 0.706 |
| 0.05 | 0.8533 | 0.9111 | 0.7857 | 0.9701 | 0.1562 | 0.256 |
| 0.10 | 0.8054 | 0.8833 | 0.7143 | 0.9431 | 0.1125 | 0.138 |
| **0.15** | 0.7844 | 0.8722 | 0.6818 | **0.9311** | **0.0813** | 0.100 |
| 0.20 | 0.7695 | 0.8556 | 0.6688 | 0.9102 | 0.0688 | 0.081 |
| 0.25 | 0.7485 | 0.8333 | 0.6494 | 0.8952 | 0.0688 | 0.075 |
| 0.30 | 0.7096 | 0.7944 | 0.6104 | 0.8383 | 0.0563 | 0.062 |
| 0.40 | 0.6677 | 0.7556 | 0.5649 | 0.7605 | 0.0312 | 0.031 |
| 0.50 | 0.6048 | 0.7056 | 0.4870 | 0.6766 | 0.0250 | 0.025 |
| 0.60 | 0.5030 | 0.6000 | 0.3896 | 0.5539 | 0.0125 | 0.013 |
| 0.70 | 0.4192 | 0.5389 | 0.2792 | 0.4551 | 0.0000 | 0.000 |
| 0.80 | 0.2455 | 0.3389 | 0.1364 | 0.2485 | 0.0000 | 0.000 |
| 0.90 | 0.0479 | 0.0833 | 0.0065 | 0.0509 | 0.0000 | 0.000 |

Two definitions of `D(t)` are carried, because they answer different questions
and the cost model cares about the second:

* **localised** -- a class-correct box on the ground-truth box at IoU >= 0.5.
  This is `src/evaluate.py`'s own definition and it measures box quality.
* **flag rate** -- the frame got a box at all, so an operator looks at it and the
  miss cost is not incurred. The cost model prices *escaped* defects, so this is
  arguably the term it wants.

Applying the identical rule to each population:

| population | detection definition | constrained t | D | F | floor met | at sweep edge | unconstrained t |
|---|---|---|---|---|---|---|---|
| pooled | localised | 0.01 | 0.9042 | 0.2812 | yes | **YES** | 0.20 |
| NEU-DET only | localised | 0.05 | 0.9111 | 0.1562 | yes | no | 0.20 |
| Severstal only | localised | 0.20 | 0.6688 | 0.0688 | **no** | no | 0.20 |
| **pooled** | **flag rate** | **0.15** | **0.9311** | **0.0813** | **yes** | **no** | **0.15** |
| NEU-DET only | flag rate | 0.20 | 0.9722 | 0.0688 | yes | no | 0.20 |
| Severstal only | flag rate | 0.05 | 0.9351 | 0.1562 | yes | no | 0.15 |

The pooled localised pick lands on 0.01, the bottom edge of the sweep. That is a
sweep-edge artefact and I am discarding it, for a stated reason: pooling two
domains of very different difficulty under one 90% floor means Severstal's harder,
diffuse, noisily-masked defects drag the pooled localised rate below the floor
everywhere except the edge. The report says so rather than quietly shipping 0.01.

Every non-degenerate criterion lands between **0.15 and 0.20**, and the pooled
flag-rate criterion -- where the constrained pick and the unconstrained cost
minimum *coincide* -- lands exactly on **0.15**.

**Recommended operating point for the joint model: conf 0.15, iou 0.45, imgsz
256.** Re-picking on validation reproduces the threshold the system already
ships, which is a convenient outcome and not one I engineered -- the number fell
out of the cost model.

Sensitivity to the miss:false-alarm ratio, which is an assumption and not a
Jindal measurement:

| miss:FA | lambda | flag-rate pick (constrained / free) | localised pick (constrained / free) |
|---|---|---|---|
| 3:1 | 0.158 | 0.20 / 0.40 | 0.01 (edge) / 0.45 |
| 10:1 | 0.526 | 0.20 / 0.20 | 0.01 (edge) / 0.20 |
| **12:1** | 0.632 | **0.15 / 0.15** | 0.01 (edge) / 0.20 |
| 30:1 | 1.579 | 0.15 / 0.15 | 0.01 (edge) / 0.03 |

Across a 10x range of the cost ratio the recommendation moves only between 0.15
and 0.20 on the flag-rate criterion, and the free optimum on the localised
criterion stays at 0.20 for 10:1 and 12:1. It only runs away at the extremes,
which is the identifiability caveat `false_alarm.py` already documents: the
decision depends on prevalence and ratio only through
lambda = prevalence x ratio / (1 - prevalence), and neither is separately
identifiable from a decision.

---

## 10. The call

**Ship the joint model as an addition, not as a silent replacement. Do not
repoint `resolve_weights` yet.** `src/inference.py` was not modified by this
work; `resolve_weights()` still returns
`models/yolov8n_neudet/weights/best.pt`, verified after the run.

### Why the joint model should replace the shipped one on the merits

Every axis is equal or better, and the one that mattered is transformed:

| axis | shipped | joint | verdict |
|---|---|---|---|
| NEU-DET test mAP50 @256 | 0.7524 | 0.7642 | **no cost** (+0.0118) |
| NEU-DET test mAP50-95 @256 | 0.3967 | 0.4008 | **no cost** (+0.0041) |
| Severstal test mAP50 | 0 by construction | 0.4559 @256 / 0.5656 @320 | new capability |
| clean-frame FA @0.15 | 93.7% | 32.5% | **-61 points** |
| clean-crop FA @0.15 | 55.1% | 5.9% | **9.3x fewer** |
| Severstal defect recall @0.15 | 72.0% | 86.6% | **+14.6 points** |
| cross-domain AUC | 0.608 [0.576, 0.640] | 0.958 [0.948, 0.968] | **chance -> usable** |
| in-domain AUC | 0.968 [0.948, 0.984] | 0.940 [0.905, 0.970] | -0.028, intervals overlap |
| clean-coil disposition @0.15 | HOLD | DOWNGRADE | **no longer held** |

There is no trade-off to present. The scenario the task anticipated -- "if the
in-domain number dropped materially, keep both and show the trade" -- did not
happen; the in-domain number did not drop.

### Why it must not be repointed today anyway

The blocker is not the model, it is the four things around it, and every one is
concrete:

1. **`src/inference.py` cannot serve a 10-class checkpoint.** `score_detection`
   indexes `DEFECT_INFO` by predicted class name and raises rather than guessing:

   ```
   KeyError: 'class_9'      src/inference.py:211, reached from predict_batch
   ```

   This is not hypothetical -- it killed the first two runs of the cross-domain
   harness in this session. `CLASS_NAMES`, `CLASS_COLORS` and `DEFECT_INFO` need
   the four Severstal entries in the file, not patched in at runtime as
   `src/train_joint.py register_severstal_classes` does for measurement.
   `src/inference.py` is not this task's file to edit, so this is handed over
   rather than done. The runtime patch is deliberately reversible and the test
   file restores every loaded copy of the module around every test -- without
   that, `tests/test_smoke.py`'s six-class contract fails, which is exactly the
   kind of breakage a silent repoint would have caused in the demo.

   > **CLOSED, by later work outside this task (verified 2026-09-10).**
   > `src/inference.py` now carries `JOINT_CLASS_NAMES`, native `DEFECT_INFO`
   > entries for `severstal_1..4` (placeholder `SEVERSTAL_TIER = "medium"`), and
   > `score_detection` degrades to `UNKNOWN_CLASS_TIER` instead of raising. Loading
   > `models/yolov8n_joint/weights/best.pt` through a plain `DefectDetector` --
   > no monkeypatch, no `register_severstal_classes` -- now returns
   > `severstal_1..4` detections with a severity and no exception; re-checked live
   > during this verification pass. **Item 1 of the four blockers below is
   > resolved.** Items 2-4 are not: `SEVERSTAL_TIER` is still the neutral
   > placeholder `"medium"` (item 2), the dual-import hazard in
   > `evaluate.py`/`false_alarm.py`/`calibrate.py` is unchanged (item 3, confirmed
   > by identity check `src.inference is inference -> False`), and
   > `demo/app.py:preferred_checkpoint_index` still hard-codes `yolov8n_neudet` /
   > `yolov8s_neudet` and `false_alarm.py`/`export_model.py` still assume
   > `len(CLASS_NAMES) == 6` (item 4, confirmed by re-reading those files). The
   > recommendation in this section -- ship alongside, do not repoint yet -- is
   > therefore still correct, for three reasons instead of four.

2. **Nobody has set a severity tier for the four Severstal classes**, and section
   8 shows the coil verdict swings ACCEPT / DOWNGRADE / HOLD on that choice
   alone. Inventing the tier would fabricate a quality judgement. This is a
   quality-department decision and it gates the disposition claim.

3. **A dual-import hazard is now load-bearing.** `evaluate.py`, `false_alarm.py`
   and `calibrate.py` each carry a `try: from .inference ... except ImportError:
   from inference ...` fallback, and one except-branch also puts `src/` on
   `sys.path`. A single process can therefore end up holding *two* module objects
   for `src/inference.py`, with two independent copies of `CLASS_NAMES` and
   `DEFECT_INFO`. That is invisible today because the six names never change; it
   will not stay invisible the moment they do. It cost real debugging time in
   this session and it should be collapsed to one import path.

4. **`demo/app.py` has not been checked against a 10-class head** by me -- another
   agent owns it, and the demo's class filters, colour legend and defect
   explainer pages all key off `CLASS_NAMES` / `DEFECT_INFO`.

### The recommendation, in one line

Ship `models/yolov8n_joint/weights/best.pt` **alongside** the current default at
conf 0.15 / iou 0.45 / imgsz 256, present it as the cross-domain answer with the
numbers in section 7c, and make repointing `resolve_weights` the *next* task --
one that starts by adding the four classes to `src/inference.py`, collapsing the
dual-import fallback, and getting a tier for `severstal_1..4` from quality. On
the evidence the switch is right; on the plumbing it is not a one-line change,
and doing it silently inside this task would have broken the demo.

---

## 11. What this does and does not prove

* Severstal is hot-rolled carbon strip photographed on someone else's line. **It
  is not Jindal stainless.** A good score here does not prove the model works at
  Hisar. The asymmetry `cross_domain.py` is built on still holds: a bad score
  disproves transfer, a good score only removes the disproof.
* The in-domain gain (+0.0118 mAP50) is one run against one run on 180 images
  with no paired bootstrap. Read it as "no measurable cost", not as an
  improvement.
* The in-domain AUC drop (0.968 -> 0.940) has overlapping confidence intervals
  and is not established. It is also not zero, and it is the one number in this
  report that moved the wrong way.
* Severstal's masks are imperfect and its four classes are unnamed. The
  cross-domain recall numbers inherit that label noise.
* 51% of the available Severstal training crops were deliberately discarded to
  respect the harness's holdout rule. A model trained on all of them would very
  likely be better and would not be measurable by this harness. That is the
  correct trade and it is the reason these numbers can be trusted.
* The clean arm of the cross-domain population is Severstal's own
  `has_defect = False` subset. If that flag is wrong on some frames, the false
  alarm rates here are pessimistic, not optimistic.

## 12. Every command in this report, in order

```bash
cd .
PY=.venv/bin/python
SAMPLES=<audit scratch>/samples.json          # Voxel51/severstal_steel_defects metadata

# 1. verify the dataset before training on it
$PY -m src.train_joint verify --samples-json $SAMPLES \
    --json reports/gap1_dataset_verification.json

# 2. derive the holdout-safe training view (symlinks; data/joint is not written to)
$PY -m src.train_joint build

# 3. widen the shipped 6-class head to 10 classes, keeping channels 0-5
$PY -m src.train_joint warmstart

# 4. train  (200 epochs, patience 40, imgsz 320, batch 32; 2 h 34 m of compute)
$PY -m src.train_joint train --epochs 200 --patience 40 \
    --imgsz 320 --batch 32 --name yolov8n_joint

# 5a/5b. NEU-DET and Severstal held-out test metrics, both checkpoints
$PY -m src.train_joint eval --weights models/yolov8n_joint/weights/best.pt \
    --json reports/gap1_detection_metrics.json
$PY -m src.train_joint eval --weights models/yolov8n_neudet/weights/best.pt \
    --json reports/gap1_detection_metrics_baseline.json

# 5c. cross-domain harness + clean-coil disposition, identical protocol to the baseline
cp reports/cross_domain.json reports/gap1_cross_domain.json   # seed with the baseline record
$PY -m src.train_joint crossdomain -- \
    --weights models/yolov8n_joint/weights/best.pt --tag joint \
    --json reports/gap1_cross_domain.json \
    --md reports/gap1_cross_domain.md \
    --plot reports/gap1_cross_domain_fa_vs_recall.png

# 6. the disposition chain on the identical 480 clean tiles, both models, several thresholds
for CONF in 0.15 0.25 0.40 0.50; do
  $PY -m src.report --images reports/cross_domain_coil/joint \
      --weights models/yolov8n_neudet/weights/best.pt \
      --conf $CONF --imgsz 256 --no-cam --out-dir /tmp --stem base_$CONF
  $PY -m src.train_joint coil --tier medium -- \
      --images reports/cross_domain_coil/joint \
      --weights models/yolov8n_joint/weights/best.pt \
      --conf $CONF --imgsz 256 --no-cam --out-dir /tmp --stem joint_$CONF
done
# severity-tier sensitivity: --tier low | medium | high | critical

# 7. re-pick the operating threshold on VALIDATION
$PY -m src.train_joint threshold --weights models/yolov8n_joint/weights/best.pt \
    --json reports/gap1_operating_point.json

# 8. tests
$PY -m pytest tests/test_train_joint.py -q               # 36 passed
$PY -m pytest tests/test_smoke.py -q                     # 149 passed, unchanged
$PY -m pytest tests/test_smoke.py tests/test_train_joint.py -q   # 185, both orders
$PY -m pytest tests/ -q                                  # 388 passed
```

### Artefacts

| path | what |
|---|---|
| `models/yolov8n_joint/weights/best.pt` | the joint checkpoint, SHA-256 `198e9b4bec5b4809...` |
| `models/yolov8n_joint/warmstart_10cls.pt` | the widened, untrained checkpoint (section 5) |
| `models/yolov8n_joint/{results.csv,args.yaml,train_console.log,train_result.json}` | the run |
| `data/joint_xdsafe/` | holdout-safe training view (symlinks) + `manifest.json` |
| `reports/gap1_dataset_verification.json` | every check in section 3 |
| `reports/gap1_detection_metrics{,_baseline}.json` | axes 7a and 7b |
| `reports/gap1_cross_domain.{json,md}`, `reports/gap1_cross_domain_fa_vs_recall.png` | axis 7c, both tags |
| `reports/cross_domain_clean_coil_joint.{html,json}` | the clean-coil report, section 8 |
| `reports/gap1_operating_point.json` | the full validation sweep and all six picks |
| `src/train_joint.py`, `tests/test_train_joint.py` | the code and its 36 tests |
