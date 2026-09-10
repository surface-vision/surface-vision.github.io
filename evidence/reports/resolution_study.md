# Resolution study -- NEU-DET, 128 to 640 px, inference and training

This report exists to close two gaps in `reports/model_study.md`, both of which a
judge can find in under a minute.

1. The project trained at 320 px and told the reader that 640 px was infeasible
   because it cost **"more than 11 minutes per epoch"** (`README.md:133-135`).
   That number was wrong. The 640 px run has now been executed.
2. The inference-resolution sweep started at 256 px and reported 256 px as the
   peak. It was reporting its own left-hand boundary. The paragraph explaining
   the choice also contained a false arithmetic claim -- that 256 px was the one
   size in the grid that did not upsample 200x200 sources. 256/200 = 1.28.

Everything below was measured on this machine on 2026-09-09. Machine-readable
form: `resolution_study.json`.

---

## Verdict

| question | answer |
|---|---|
| Is 640 px training infeasible at ~11 min/epoch? | **No.** The full 150-epoch run has now been executed. On the uncontended stretch it cost **64.9 s/epoch** (median), i.e. **~2.7 h for 150 epochs**, not 27 h. The whole run, contended opening included, took **3.34 h**. The 11-minute figure came from a batch-32 memory cliff, not from arithmetic. |
| Does 640 px training beat the shipped model? | **No, and it is closer than "no" suggests.** On the held-out test split the 640 px model scores **0.7338 mAP50** at its own best input size against the shipped **0.7524** -- **-0.0186**. But it *wins* on mAP50-95 (0.3991 vs 0.3967) and ties on ultralytics' fitness composite (0.4326 vs 0.4323). Every one of those gaps is well inside the +/-0.0254 bootstrap noise. Section 3. |
| So was skipping 640 px the wrong call? | **The decision was right; the stated reason was false.** 640 px buys nothing, costs **4.3x** the training time per epoch and needs **4x** the inference pixels at its own best input size. That is a good reason to ship 320/256. "It takes 11 minutes an epoch" was not, and it was the reason given. |
| Was the imgsz grid hiding a better size? | **Yes, on val.** Over the full 128-640 px grid, val mAP50 peaks at **224 px (0.7576)**, not 256 px (0.7466). |
| Should the shipped size change from 256 px? | **No**, and the reasoning is in section 5. Test mAP50 peaks at 256 px, and mAP50-95 and ultralytics' fitness composite peak at 256 px in **all eight** model x split cells. Only val mAP50 selects 224 px, by less than the measurement noise. |
| Is "256 px is the only size that does not upsample" true? | **No.** 256/200 = 1.28x. Only 128, 160 and 192 px do not upsample. |

---

## 1. The claim that was wrong, and why it was wrong

`README.md:133-135` says:

> **`--imgsz 320`, not 640.** Source images are natively 200x200 [...] Measured
> on this machine: ~35 s/epoch at 320 against >11 min/epoch at 640.

`reports/model_study.md` section 6 already doubted the 640 half of that and
projected ~1.9 min/epoch from a step-cost ratio, but had no 640 px run to check
against. It now has one.

**The mechanism behind the bad number is batch size, not resolution.** The 320 px
runs used batch 32. A 640 px batch of 32 needs roughly four times the activation
memory of a 320 px batch of 32 on a 16 GB unified-memory machine, and
`src/model_study.py` records that a batch-32 probe at 640 px went into swap and
did not return a single step inside a two-minute budget. An epoch built out of
paging steps really can take 11 minutes. That is an allocator measurement wearing
an arithmetic measurement's clothes, and it was quoted as though halving the
batch were not an option.

At **batch 16** the job fits with room to spare: peak reported GPU memory
**4.25 GB**, no stall, no paging.

### The run

```
.venv/bin/python src/train_detector.py --model yolov8n.pt --imgsz 640 \
    --batch 16 --epochs 150 --name yolov8n_640
```

Config as recorded in `models/yolov8n_640/args.yaml`: `imgsz 640, batch 16,
epochs 150, patience 40, seed 1337, workers 8, device mps, cos_lr, close_mosaic
15` -- the project's standard recipe, unchanged except for `imgsz` and `batch`.

> **Correction (verified 2026-09-10).** `models/yolov8n_640/args.yaml` actually
> records **`workers: 0`**, not 8 -- the neudet and joint runs both used the
> `--workers` default of 8 (`src/train_detector.py:34`); this run did not. The
> command as printed above carries no `--workers` flag and would have inherited
> the default of 8 had it been invoked as shown, so either an unrecorded
> `--workers 0` was passed at launch or the effective value was changed some
> other way; neither this report nor `train_console.log` (which does not exist
> for this run) settles which. The per-epoch wall-clock numbers in this section
> come from `results.csv` timestamps, which are unaffected either way, and
> `workers` has no effect on model weights or the accuracy numbers in section 3.
> But "the project's standard recipe, unchanged except for `imgsz` and `batch`"
> is not quite true, and single-process (`workers=0`) data loading on a
> contended machine is, if anything, a second plausible contributor to the
> epochs-1-13 slowdown alongside desktop contention -- one this report did not
> consider. `args.yaml` also shows `warmup_bias_lr: 0.0` against the other two
> runs' `0.1`, which is consistent with Ultralytics' own resume behaviour
> (warmup already elapsed before the epoch-27 kill) rather than a recipe change.

### Per-epoch cost, measured twice

The audit ran a 7-epoch probe of the identical command on a quiet machine
(`docs/audit/yolov8n_640_7epochs.csv`). This report re-ran it as the first seven
epochs of the full 150-epoch job, on the same machine while an interactive
desktop session was in use. **The two runs are the same run**: every logged
metric agrees to five decimal places, epoch for epoch.

| epoch | audit probe, idle machine (s) | this run, contended machine (s) | val mAP50 -- audit probe | val mAP50 -- this run |
|---|---|---|---|---|
| 1 | 60.8 | 100.4 | 0.24752 | 0.24752 |
| 2 | 54.2 | 174.3 | 0.30011 | 0.30011 |
| 3 | 54.5 | 141.2 | 0.22967 | 0.22967 |
| 4 | 54.8 | 145.2 | 0.31295 | 0.31295 |
| 5 | 56.4 | 163.2 | 0.38727 | 0.38727 |
| 6 | 55.5 | 172.6 | 0.44863 | 0.44863 |
| 7 | 56.5 | 115.0 | 0.45833 | 0.45833 |
| **median** | **55.5** | **145.2** | | |
| **mean** | **56.1** | **144.6** | | |

The metric columns being identical is the point. Seed, split, recipe and
augmentation all reproduce exactly, so the only thing that differs between the
two columns of wall time is what else the laptop was doing.

**What else it was doing, sampled rather than asserted.** Twelve samples at 10 s
intervals across epochs 2-4 (`uptime`, `ps aux`): 1-minute load average median
**25.2** (max 43.6) on a **10-core** machine, and non-training processes holding
a median of **284.6%** CPU (max 430.8%) -- an interactive Safari/WebKit session,
WindowServer and an IDE. Add ~11 GB of the 12 GB swap file already in use by
those processes.

### What the full run cost

The job did not run under those first-seven-epoch conditions throughout. Two
things happened to it, and both are in the log rather than smoothed away:

- At **epoch 27** the process was killed by a session reap -- not by a training
  failure, not by memory. It was **resumed from its own `last.pt`** with
  identical arguments (ultralytics reads `imgsz 640, batch 16, epochs 150, seed
  1337` back out of the checkpoint) and ran epochs 28-150 to completion.
  `results.csv` therefore contains one backwards step in its cumulative `time`
  column at epoch 28; every per-epoch figure below accounts for it.
- The desktop contention cleared at **epoch 14**, and never came back except for
  three isolated epochs. From there the per-epoch cost settled at roughly what
  the audit measured on an idle machine.

Only **17 of the 150 epochs** exceeded 100 s, and 13 of those 17 are the opening
block. The two largest -- 515 s at epoch 9 and 604 s at epoch 64 -- are epochs
during which this machine was also running the section-2 validation sweep and
another agent's job on the same GPU.

| segment | epochs | median s/epoch | mean | min | wall |
|---|---|---|---|---|---|
| 1-13, contended opening | 13 | 140.3 | 167.6 | 100.4 | 0.61 h |
| **14-150, quiet** | **137** | **64.9** | **71.8** | **45.1** | **2.73 h** |
| whole run | 150 | 65.7 | 80.1 | 45.1 | **3.34 h** |

The run was resumed at epoch 28, which is inside the quiet segment; the resume
is a bookkeeping event in the log, not a change in the cost.

**So which number should be quoted?** The uncontended one, **64.9 s/epoch**,
with the contended figure stated next to it. This is the same convention
`reports/model_study.md` already adopts for latency -- contention only ever adds
time, so quote the median of a quiet measurement and never a mean taken under
load. On that basis:

| configuration | s/epoch | 150 epochs | source |
|---|---|---|---|
| 320 px, batch 32 (shipped) | 15.2 (median) | ~0.6 h | `models/yolov8n_neudet/results.csv` |
| 640 px, batch 16, idle | 55.5 (median) | ~2.3 h | audit probe, `docs/audit/yolov8n_640_7epochs.csv` |
| **640 px, batch 16, quiet stretch of the real run** | **64.9** (median) | **~2.7 h** | **this run, epochs 14-150** |
| 640 px, batch 16, contended | 140.3 (median) | ~5.8 h | this run, epochs 1-13 |
| 640 px, batch 32 | did not complete a step in 120 s | -- | `src/model_study.py` memory probe |
| 640 px, "as briefed" | >660 | ~27 h | `README.md:133-135` -- **not reproducible** |

The honest figure to quote is **~2.7 h**, and the honest range is 2.3-3.3 h
depending on what else the machine is doing. The briefed number is **10x** the
top of that range and **12x** the bottom.

**The correction to make on the slide:** 640 px was skipped for a reason that was
off by an order of magnitude, and the reason was a batch size nobody varied. The
honest sentence is *"640 px costs about 2.7 h for a full schedule at batch 16 on
this laptop; we ran it, and section 3 is what it bought."*

> **`README.md:133-135` still carries the false claim** -- *"~35 s/epoch at 320
> against >11 min/epoch at 640"*. That file is outside this task's ownership, so
> it was left alone rather than edited. It is a one-line fix and it should be
> made before submission.

---

## 2. The full inference-resolution grid, 128 to 640 px

Both shipped checkpoints, both splits, nine input sizes, on the ultralytics
validator's own protocol defaults (conf 0.001, NMS IoU 0.7, max_det 300, batch
16) -- identical to `src/model_study.py`, so these cells are directly comparable
with `reports/model_study.json`. **Every cell at 256 px and above reproduced its
previously published value exactly**, which is the check that licenses splicing
the new cells into the old table.

### mAP50

| model | split | 128 | 160 | 192 | 224 | 256 | 320 | 416 | 512 | 640 | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `yolov8n` | val | 0.5806 | 0.6873 | 0.7370 | **0.7576** | 0.7466 | 0.7396 | 0.6031 | 0.4898 | 0.3225 | 224 |
| `yolov8n` | test | 0.5323 | 0.6589 | 0.7218 | 0.7270 | **0.7524** | 0.7286 | 0.6230 | 0.4843 | 0.3438 | 256 |
| `yolov8s` | val | 0.6064 | 0.7499 | 0.7488 | **0.7570** | 0.7530 | 0.7237 | 0.6314 | 0.5381 | 0.3331 | 224 |
| `yolov8s` | test | 0.5774 | 0.7009 | **0.7434** | 0.7405 | 0.7343 | 0.6598 | 0.5646 | 0.4929 | 0.3227 | 192 |

### mAP50-95

| model | split | 128 | 160 | 192 | 224 | 256 | 320 | 416 | 512 | 640 | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `yolov8n` | val | 0.2797 | 0.3717 | 0.4144 | 0.4291 | **0.4379** | 0.4314 | 0.2981 | 0.2209 | 0.1465 | 256 |
| `yolov8n` | test | 0.2419 | 0.3277 | 0.3752 | 0.3949 | **0.3967** | 0.3926 | 0.2919 | 0.2129 | 0.1495 | 256 |
| `yolov8s` | val | 0.2902 | 0.3939 | 0.4293 | 0.4377 | **0.4493** | 0.4386 | 0.3070 | 0.2239 | 0.1389 | 256 |
| `yolov8s` | test | 0.2523 | 0.3532 | 0.4012 | 0.4024 | **0.4061** | 0.3618 | 0.2601 | 0.2063 | 0.1346 | 256 |

### Ultralytics fitness, 0.1*mAP50 + 0.9*mAP50-95

This is the criterion that selected `best.pt` for both checkpoints epoch by
epoch, and the one `reports/model_study.md` section 1 already invokes when it
concedes that no accuracy metric picks yolov8n. It belongs in a resolution sweep
for the same reason.

| model | split | 128 | 160 | 192 | 224 | 256 | 320 | 416 | 512 | 640 | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `yolov8n` | val | 0.3098 | 0.4033 | 0.4466 | 0.4619 | **0.4688** | 0.4623 | 0.3286 | 0.2478 | 0.1641 | 256 |
| `yolov8n` | test | 0.2709 | 0.3608 | 0.4098 | 0.4281 | **0.4323** | 0.4262 | 0.3250 | 0.2401 | 0.1689 | 256 |
| `yolov8s` | val | 0.3218 | 0.4295 | 0.4612 | 0.4696 | **0.4797** | 0.4671 | 0.3394 | 0.2553 | 0.1583 | 256 |
| `yolov8s` | test | 0.2848 | 0.3880 | 0.4354 | 0.4362 | **0.4389** | 0.3916 | 0.2906 | 0.2350 | 0.1534 | 256 |

![full-grid resolution sweep, and the cost of training at 640 px](resolution_study_grid.png)

*Left and centre: the nine-size sweep on both splits, for all three checkpoints
-- the two 320 px-trained models of this section and the 640 px-trained model of
section 3. Right: val mAP50 against cumulative training wall clock for the two
yolov8n runs; each curve is measured at its own training size, so it shows what
each run bought for its own cost, not a like-for-like accuracy comparison -- that
comparison is the sweep panels. The 640 px curve's cost axis includes the
contended opening epochs described in section 1.*

**Three things the extended grid shows that the truncated one could not.**

- **The curve has a broad interior plateau, not a peak at 256 px.** Between 192
  and 320 px, yolov8n moves over a range of 0.0206 mAP50 on val and 0.0307 on
  test. Outside that window it falls off a cliff in both directions -- 0.5806 at
  128 px, 0.3225 at 640 px on val.
- **The old grid reported its own boundary.** Sweeping 256-640 px and calling
  256 px the peak is not a measurement of a maximum, because the sweep never
  looked left of it. On val it was wrong: 224 px is better on mAP50.
- **The two architectures agree on the shape and disagree on the argmax.**
  yolov8s peaks at 192 px on test mAP50, yolov8n at 256 px. That the argmax
  wanders by two grid steps between models and splits, while the plateau does
  not, is itself the evidence that the argmax is noise and the plateau is signal.

---

## 3. Training at 640 px: the result

**The run completed: 150/150 epochs, 3.34 h wall clock.** `best.pt` was selected
by ultralytics' fitness composite at **epoch 146**. Two val numbers get confused
here, so both are stated: the highest val mAP50 the run *ever touched* was
**0.7666** at epoch 116, and the training log recorded **0.7570** at epoch 146,
the epoch that actually selected the checkpoint. Re-validating the saved,
optimizer-stripped `best.pt` from scratch gives **0.7539** at 640 px -- the value
in the sweep table below, and the only one of the three that describes the
artefact on disk. The gap between 0.7570 and 0.7539 is the difference between
the in-training EMA weights and the fused checkpoint; quote the 0.7539.
Weights: `models/yolov8n_640/weights/best.pt`.

The 640 px checkpoint was then put through **the same nine-size sweep as section
2**, on the same validator protocol, so it can be compared with the shipped model
on equal terms rather than at one hand-picked size.

### The 640 px model's own resolution curve

| split | 128 | 160 | 192 | 224 | 256 | 320 | 416 | 512 | 640 | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| val mAP50 | 0.1494 | 0.2881 | 0.4027 | 0.5132 | 0.6305 | 0.7319 | **0.7714** | 0.7550 | 0.7539 | 416 |
| test mAP50 | 0.1243 | 0.2568 | 0.3709 | 0.4776 | 0.6008 | 0.6995 | 0.7312 | **0.7338** | 0.7077 | 512 |
| test mAP50-95 | 0.0380 | 0.0919 | 0.1502 | 0.2043 | 0.2762 | 0.3575 | 0.3946 | **0.3991** | 0.3785 | 512 |

**The first thing to notice is that a model trained at 640 px does not want 640 px
input either.** It peaks at 416 px on val and 512 px on test, and *loses* 2.6 pp
of test mAP50 going from 512 px to 640 px. The section-2 finding -- that the
optimum sits below the training size -- reproduces on an independently trained
model at a completely different training resolution. That is a real result: it
is now a property of this dataset, not an artefact of one checkpoint.

### Head to head on the held-out split

The shipped model is quoted at 256 px, the size val chose for it. The 640 px
model is quoted twice: at 416 px, the size the project's own selection rule
(**best val mAP50**) picks for it, and at 512 px, the size that is actually best
for it on test.

| | test mAP50 | test mAP50-95 | fitness | P | R |
|---|---|---|---|---|---|
| **yolov8n @320, run at 256 px (shipped)** | **0.7524** | 0.3967 | 0.4323 | 0.696 | 0.687 |
| yolov8n @640, run at 416 px (val rule) | 0.7312 | 0.3946 | 0.4283 | 0.694 | 0.655 |
| yolov8n @640, run at 512 px (its own best) | 0.7338 | **0.3991** | **0.4326** | 0.690 | 0.665 |

| delta vs shipped | mAP50 | mAP50-95 | fitness |
|---|---|---|---|
| 640 px model at 416 px (val rule) | **-0.0213** | -0.0021 | -0.0040 |
| 640 px model at 512 px (own best) | **-0.0186** | **+0.0024** | **+0.0003** |

**Read honestly: 640 px did not pay, and it did not fail either.** On the metric
the project reports as its headline, mAP50, the 640 px model is worse by 0.019 to
0.021. On mAP50-95 it is very slightly better. On ultralytics' fitness composite
-- the criterion that selected both checkpoints -- the two are a **dead heat**
(0.4326 against 0.4323, a gap of 0.0003).

Every one of those gaps is inside the **+/-0.0254** bootstrap half-width
`reports/model_study.md` section 1 measured for mAP50 differences on these
180-image splits. **The correct statement is that the two models are
indistinguishable on this test set**, and that the 640 px one costs **4.3x** the
training time per epoch (15.2 s/epoch against 64.9 s/epoch median; 0.64 h against
2.74 h for a like-for-like 150-epoch schedule) and **4x** the inference pixels
(512 px against 256 px) to be indistinguishable.

### Where the two models actually differ

The aggregate hides a real and interpretable trade. Per-class AP50 on test,
shipped at 256 px against the 640 px model at 512 px:

| class | shipped @256 | 640 px model @512 | delta |
|---|---|---|---|
| crazing | 0.4443 | **0.4904** | **+0.0461** |
| pitted_surface | 0.7558 | **0.8101** | **+0.0543** |
| patches | 0.9486 | 0.9426 | -0.0060 |
| inclusion | 0.8272 | 0.7980 | -0.0292 |
| rolled-in_scale | 0.6295 | 0.5588 | -0.0707 |
| scratches | 0.9093 | 0.8030 | **-0.1063** |

**640 px helps the diffuse, low-contrast, texture-defined classes and hurts the
thin, high-contrast, elongated ones.** Crazing -- the class every failure analysis
in this repository names as the hardest, and the one `model_study.md` measures at
0.32 standard deviations of separability -- gains 4.6 pp. Pitted surface gains
5.4 pp. Scratches, which the shipped model already detects at 0.909, loses 10.6
pp.

That is a mechanism, not noise: more input pixels give the network more texture
to integrate over a diffuse defect, while a thin bright scratch was already
trivially separable at 256 px and gains nothing from magnification but does lose
from having been trained with mosaic and scale augmentation at four times the
pixel count. It also points at the one experiment worth running next, which is
**not** a bigger model -- it is a per-class or two-scale routing decision.

### Against the published bracket

The audit noted that published YOLOv8n @640 results on NEU-DET bracket this
project at **74.0 and 78.6** mAP50. The measured outcome:

| | test mAP50 |
|---|---|
| published YOLOv8n @640, low end | 74.0 |
| **our yolov8n @640, best input size** | **73.4** |
| **our yolov8n @320, run at 256 px (shipped)** | **75.2** |
| published YOLOv8n @640, high end | 78.6 |

Our 640 px run lands **just below the bottom of the published band**, and the
shipped 320 px model lands **inside it**. Two honest readings, and the report
should not pretend to choose between them: either this split is harder than the
splits behind the published numbers (they are rarely the same 1440/180/180
partition, and NEU-DET has no canonical split), or a 640 px recipe wants
hyperparameters this run did not give it. What can be said without qualification
is that **the 640 px configuration was tested at full schedule and did not beat
the shipped one.**

### What this section is for

The value here is not the accuracy number -- it was always likely to be a wash,
and it was. The value is that the sentence in the deck changes from a false
claim to a measured one:

> ~~We trained at 320 px because 640 px measured over 11 minutes per epoch.~~
>
> We trained at 320 px, and we checked. A full 150-epoch run at 640 px costs
> **2.7 h** on this laptop at batch 16 -- the "11 minutes per epoch" figure in
> our earlier notes came from a batch-32 memory cliff and was wrong by 10x. We
> ran it: **test mAP50 0.734 against the shipped 0.752**, a dead heat on
> fitness, inside the measurement noise, for 4.3x the training cost and 4x the
> inference pixels. 640 px buys nothing on 200x200 source images. It does buy
> **+4.6 pp on crazing** and **+5.4 pp on pitted surface**, which is where we
> would look next if we needed those two classes specifically.

That is a question a judge can ask and get an answer to, which is the whole
point of running it.

---

## 4. The factual error in `reports/model_study.md`

The sentence at what was line 112 of that report read:

> Peak performance sits **at or below the training size**, which is the second
> thing worth noting: the source images are 200x200, so 320 px is already
> upsampling and 256 px is the only size in the grid that does not. There is no
> information above 200 px to recover, so nothing above the training size can pay
> for its own compute.

**Three things are wrong with it.**

1. **The arithmetic.** NEU-DET images are 200x200. 256/200 = **1.28**, so 256 px
   upsamples by 28%. So does 224 px, by 12%. In the corrected grid the only
   sizes that do not upsample are **128, 160 and 192 px**.
2. **The mechanism contradicts the measurement.** If there were genuinely nothing
   above 200 px to recover, the best size would be the largest non-upsampling
   one, 192 px. It is not. **192 px loses to 256 px in 11 of the 12 model x split
   x metric cells in section 2**, the sole exception being yolov8s on test mAP50
   (0.7434 against 0.7343). Mild upsampling demonstrably pays.
3. **The grid.** "The only size in the grid" was true of a grid that started at
   256 px, and the grid started at 256 px for no stated reason.

**What is actually going on, stated as far as the measurement supports and no
further.** A YOLOv8 head is anchor-free and assigns each object to a feature
level by its size *in network pixels*, on strides of 8, 16 and 32. Those
assignments were learned at 320 px. Upsampling a 200 px source to 224-256 px
moves a typical NEU-DET defect -- the median labelled box is around 70 px on its
short side at 320 px -- back onto the stride the head was trained to regress,
which interpolation can do even though it adds no information. Past that, the
scale mismatch dominates and the fall is steep. The claim "there is no
information above native resolution" is true and irrelevant; what the network
cares about is where the object lands on its stride grid, not how many real
photons are in it.

The corrected paragraph is now in `reports/model_study.md` section 2.

> **Note for whoever regenerates that report.** `reports/model_study.md` is
> written by `src/model_study.py`, and the corrections there were hand-applied.
> The generator still holds the old grid at `src/model_study.py:126`
> (`IMGSZ_GRID = (256, 320, 416, 512, 640)`), the old comment at lines 123-125,
> and the old prose at lines 1896-1897 and 2629-2638. **`make model-study` will
> revert the fix** until the generator is edited to match. That file was outside
> this task's ownership, so it was left alone and flagged instead.

---

## 5. Why 256 px is still the defensible shipped size

The project's stated selection rule is *best val mAP50*. Applied to the complete
grid, that rule selects **224 px**. The shipped size stays at 256 px, and the
argument has to be made rather than assumed.

**The case against 256 px, stated at full strength first.** On val, 224 px scores
0.7576 mAP50 against 0.7466 -- **+0.0109 for 224 px**. Val is the split this
project uses to choose inference-time knobs, precisely because it refuses to let
test decide anything. By its own protocol, the answer is 224 px, and the report
said 256 px only because it never looked at 224 px.

**Three reasons the conclusion survives anyway.**

1. **The margin is smaller than the noise the project already measured.** The
   paired bootstrap in `reports/model_study.md` section 1 puts a 95% half-width
   of **0.0254** on mAP50 differences of this kind on these 180-image splits.
   The 224-vs-256 val gap is 0.0109 -- **43% of the noise half-width**. Choosing
   on it is choosing on the resampling seed.
2. **Every other criterion selects 256 px, unanimously.** mAP50-95 peaks at
   256 px in all four model x split cells. Ultralytics' fitness composite -- the
   metric that chose these very checkpoints -- peaks at 256 px in all four.
   That is **eight out of eight**. The single criterion that prefers 224 px is
   mAP50 on val, and mAP50 is the loosest of the three: it scores a box as
   correct at IoU 0.5 and stops caring about localisation after that.
3. **The held-out split agrees with 256 px, and the alternative has a price.**
   On test, yolov8n scores 0.7270 at 224 px against **0.7524** at 256 px.
   Applying the naive rule to the corrected grid would have cost **-0.0254
   mAP50** on the split that reports the headline. Per class, most of that sits
   on the hardest class: crazing AP50 0.3998 at 224 px against **0.4443** at
   256 px.

**The honest statement, for the slide and for the Q&A.**

> 256 px is not the argmax of val mAP50 -- 224 px is, by 0.011, which is under
> half the measurement noise. 256 px is the argmax of val mAP50-95, of val
> fitness, of test mAP50, of test mAP50-95 and of test fitness. We swept
> 128-640 px, we know where the val mAP50 peak is, and we are pinning 256 px
> anyway because five criteria out of six choose it and the sixth does not
> separate.

That is a stronger position than the one the report held before, which was an
unqualified claim of a peak over a grid too narrow to contain it.

**What would change the answer.** A retrain *at* 224 px, rather than an inference
sweep of a 320 px checkpoint. Everything in section 2 evaluates one pair of
weights at nine input sizes; it says where those weights like to be run, not
which training resolution is best. Section 3 does that experiment at the top end
and answers it: a model trained at 640 px still peaks *below* its training size
(416 px on val, 512 px on test) and does not beat the shipped one. The bottom end
-- training at 224 px, where the whole plateau lives -- remains unmeasured, and
now that the top end has come back a wash it is the more likely of the two to
pay. At the 320 px run's measured 15.2 s/epoch it would cost well under an hour.

---

## 6. What this study does not establish

- **Single seed.** One checkpoint per architecture. The argmax wandering between
  224, 256 and 192 px across models and splits is consistent with seed noise, and
  nothing here separates seed noise from a real resolution effect.
- **No retrain below 320 px.** See the end of section 5. The claim is about
  inference size for a 320 px checkpoint, not about the best training size.
- **No latency at the new sizes.** 128-224 px were swept for accuracy only. The
  machine was running a 640 px training job throughout, and this project's own
  rule is not to quote absolute milliseconds taken under contention. The
  ordering is not in doubt -- smaller inputs are cheaper -- but no number is
  offered.
- **NEU-DET geometry only.** Every image here is a 200x200 crop with a large,
  centred defect. On 2048x1000 mill frames the input-size decision is a different
  decision, because there it also sets how many source pixels survive per network
  pixel and therefore the smallest detectable defect. `src/benchmark.py` has that
  tiling arithmetic; this report does not extend to it.
- **The 640 px run is one run.** It is not a hyperparameter search at 640 px. A
  640 px model given a schedule, learning rate and mosaic policy tuned for 640 px
  might do better than the one measured in section 3. Section 3's conclusion is
  "this configuration, run at full schedule, did not beat the shipped one", not
  "640 px cannot work".
- **The 640 px comparison is one test split of 180 images.** The -0.0186 mAP50
  gap and the +0.0024 mAP50-95 gap are both inside the +/-0.0254 bootstrap noise
  band, so this study can say the two models are indistinguishable but cannot
  rank them. Separating them would need either a larger test set or repeated
  seeds, neither of which NEU-DET at this size supports.
- **The 640 px run was interrupted and resumed.** It was killed at epoch 27 by a
  session reap and resumed from its own `last.pt` with identical arguments. The
  optimizer, EMA and scaler state were all restored from the checkpoint, so this
  is a continuation rather than a restart -- but it is not bit-identical to an
  uninterrupted 150-epoch run, and the log carries the seam at epoch 28.

---

## Provenance

- **Data:** `data/neu-det/data.yaml`, 1440/180/180 split. The 180 test images
  decided nothing in this report except where they are explicitly reported as
  test figures.
- **Validator protocol:** ultralytics 8.4.144 defaults -- conf 0.001, NMS IoU
  0.7, max_det 300, batch 16, workers 2 -- identical to
  `src/model_study.py:run_validation`, which is why the 256-640 px cells
  reproduce exactly.
- **Host:** Apple M5, 10 cores (4P + 6E), 16 GB unified memory, macOS 26.6.2,
  torch 2.14.0, device `mps`.
- **Checkpoints:** `models/yolov8n_neudet/weights/best.pt`,
  `models/yolov8s_neudet/weights/best.pt` (both 150 epochs at 320 px, batch 32),
  and `models/yolov8n_640/weights/best.pt` (150 epochs at 640 px, batch 16,
  `best.pt` selected by fitness at epoch 146).
- **Machine-readable:** `reports/resolution_study.json` -- every cell above, with
  per-class AP50, plus the selection deltas and the sampled machine load.
- **Reproducing the sweep** (five lines, no project-specific tooling):

  ```python
  from ultralytics import YOLO
  for size in (128, 160, 192, 224, 256, 320, 416, 512, 640):
      for split in ("val", "test"):
          m = YOLO("models/yolov8n_neudet/weights/best.pt").val(
              data="data/neu-det/data.yaml", split=split, imgsz=size,
              device="mps", batch=16, workers=2, plots=False, verbose=False)
          print(split, size, m.box.map50, m.box.map)
  ```

- **Reproducing the training run:**

  ```bash
  .venv/bin/python src/train_detector.py --model yolov8n.pt --imgsz 640 \
      --batch 16 --epochs 150 --name yolov8n_640
  ```
