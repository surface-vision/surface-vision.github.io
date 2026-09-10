"""Inference layer for the Jindal Stainless surface-defect detector.

This module is the single integration point between the trained YOLO detector and
everything downstream (Streamlit UI, batch scoring scripts, reporting). It owns:

* the canonical class list, display colours and mill-facing defect knowledge base,
  covering both the shipped 6-class NEU-DET head and the 10-class joint head in
  `models/yolov8n_joint` (`JOINT_CLASS_NAMES`); a detector takes its class list
  from the checkpoint it loaded, so either head reports real class names,
* a `DefectDetector` wrapper that returns a structured, JSON-serialisable
  `InferenceResult` instead of raw framework objects,
* honest per-stage latency measurement (`time.perf_counter`, device-synchronised
  -- including the framework's own stage clocks, which are not MPS-aware by
  default and would otherwise bill the forward pass to postprocess),
* sliding-window inference (`predict_tiled`) for strip images far wider than the
  network input, with a single global NMS so a defect crossing a tile seam is
  counted once -- and, at the default tile size, with no resampling anywhere in
  the path: see the lossless contract on `DefectDetector.predict_tiled`,
* a readable annotation renderer that scales with image size.

Importing this module has no side effects and prints nothing: the heavy
`ultralytics` import is deferred until a detector is actually constructed.
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/inference.py` from the Jindal Stainless surface-defect
# repository, carried into the Hugging Face Space so the app imports nothing from
# outside this directory (the repository's `src/` does not exist on the Space
# host). The upstream file is the source of truth; re-vendor rather than patch.
#
# `PROJECT_ROOT` in these modules is `Path(__file__).resolve().parent.parent`,
# which on the Space resolves to the Space root -- so `models/`, `reports/`,
# `data/` and `assets/` are the trimmed copies shipped beside this directory and
# every path the upstream code builds still resolves.
#
# Deviations from upstream are marked `# SPACE:` inline and listed in
# `vendor/README.md`. There are none in this file unless such a marker appears.
# ---------------------------------------------------------------------------
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# The network input size everything defaults to, defined here once so the console,
# the coil reporter, the exporter and the explainer cannot each pick their own.
#
# 256, not the framework's generic 640. This is an accuracy control: the shipped
# checkpoints were fine-tuned at 320 px, and measured test mAP50 runs 0.752 /
# 0.729 / 0.623 / 0.484 / 0.344 at 256 / 320 / 416 / 512 / 640 px
# (`reports/model_study.json`). 256 was chosen on val by src/model_study.py.
# Everything in this tree that used to inherit 640 was running the detector at
# less than half its accuracy.
#
# RE-OPENED TWICE AND CLOSED AGAIN, with the holes in the grid filled and the
# no-resampling alternative actually built. The full argument, and the one
# experiment that would legitimately reopen it, are in
# `reports/input_pipeline_decision.md`. In short:
#
# * `reports/input_study.md` swept 160-384 px on both checkpoints and both
#   splits. Every size from 224 to 320 px lands within 0.0254 mAP50 of its own
#   curve's best and the paired bootstrap (1000 resamples of 180 images, seed
#   1337) puts zero inside every interval in that band, so there is no evidence
#   to move off 256; the ends of the grid give up as much as 0.1075. The val
#   peak at 224 px that an audit flagged is a maximum selected from 8 sizes;
#   corrected for that it is 99.3% [-0.0057, +0.0479] and contains zero.
# * The tempting alternative -- NEU-DET is natively 200x200, so pad it into a
#   224 or 256 canvas with NO scaling and keep every source pixel byte-exact --
#   was built, verified lossless on 180/180 images, and LOSES: -0.0238 to
#   -0.0521 mAP50 on 8 of 8 configurations (`reports/input_study.md`).
# * Why it loses is the part that decides this constant, and
#   `reports/input_pipeline_probe.json` measures it. Magnify the same source to
#   256 px by nearest neighbour rather than bilinear and it is still exactly
#   lossless -- the probe recovers the 200x200 source byte-for-byte out of the
#   256x256 image by pure indexing -- yet held-out test mAP50 falls to 0.6252
#   (yolov8n_neudet) and 0.6257 (yolov8n_joint), the worst of five arms. An arm
#   that genuinely destroys 36% of the source samples (200 -> 160 -> 256) beats
#   it, at 0.6910 / 0.7296. Preserving pixels is not what buys accuracy here:
#   presenting the defect at the magnification, and through the resampling
#   kernel, that the network was trained on is. 256 px bilinear is that path.
DEFAULT_IMGSZ = 256

# Index order is fixed by data/neu-det/data.yaml and by the trained head.
#
# This is the *NEU-DET* contract and it stays six long. `data/neu-det/data.yaml`,
# the dataset integrity tests, `src/false_alarm.py`'s per-class tables and the
# atlas in the console all index it, so widening it would silently change what
# "chance is 1/len(CLASS_NAMES)" means. The 10-class joint head is described by
# JOINT_CLASS_NAMES below; a detector reads its class list off the checkpoint it
# actually loaded, never off this list.
CLASS_NAMES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# The four extra classes carried at head indices 6-9 by models/yolov8n_joint,
# which was trained on NEU-DET plus Severstal with the NEU-DET indices unchanged
# (data/joint_xdsafe/data.yaml). The names are the mask values the Severstal
# release ships, because that release publishes nothing else -- see the note on
# SEVERSTAL_TIER below.
SEVERSTAL_CLASS_NAMES: tuple[str, ...] = (
    "severstal_1",
    "severstal_2",
    "severstal_3",
    "severstal_4",
)

# The full 10-class contract of the joint checkpoint. Built once at import from a
# copy of CLASS_NAMES, so a consumer that mutates CLASS_NAMES in place cannot
# reach in here.
JOINT_CLASS_NAMES: list[str] = list(CLASS_NAMES) + list(SEVERSTAL_CLASS_NAMES)

# Distinct hues that stay legible against light grey mill imagery. All ten
# classes of the joint head are covered: an uncoloured class falls back to white
# in `annotate`, which is the one colour mill imagery is full of.
#
# The four Severstal hues were picked out of the gaps the NEU-DET six leave in
# RGB (yellow-green, pink, brown, slate) -- every new pair sits further apart
# than the tightest shipped pair, which is crazing/rolled-in_scale at 61.8.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "crazing": (255, 130, 0),
    "inclusion": (220, 30, 45),
    "patches": (55, 150, 255),
    "pitted_surface": (170, 80, 220),
    "rolled-in_scale": (240, 190, 0),
    "scratches": (0, 200, 160),
    "severstal_1": (140, 200, 60),
    "severstal_2": (255, 120, 190),
    "severstal_3": (120, 85, 40),
    "severstal_4": (70, 95, 120),
}

# Base tier for the four Severstal classes.
#
# WHAT IS KNOWN: the Severstal release publishes a numeric mask value 1-4 and
# nothing else. The competition data description names no defect type, the
# dataset card documents none, and the mask files carry only the integer. This
# was checked rather than assumed; `reports/severstal_dataset.md` records the
# same finding, which is why the classes are named after their mask value.
#
# WHAT IS NOT KNOWN, AND IS NOT GUESSED HERE: what the four defects physically
# are. A mapping onto the NEU-DET vocabulary ("class 1 is pitted surface, class
# 2 is crazing, ...") is repeated in several write-ups of the competition; it is
# unsourced, it contradicts the fact that the joint model separates those NEU
# classes from these at the same time, and it is not used. Writing a reheat
# furnace root cause for `severstal_3` would be fabrication.
#
# WHY "medium": the tier has to be something, so it is the neutral middle of
# SEVERITY_BANDS, and the consequences of that choice are bounded and stated.
# One such detection scores at most 42 * 1.0 * 1.55 = 65.1, so it can lift a
# frame into the "high" band and DOWNGRADE a coil, but never on its own into
# "critical", which is the band that forces an unconditional HOLD in
# `report.DispositionRules`. It is also not in `hold_classes`. So an unnamed
# defect can cost a coil its prime grade but cannot stop the line by itself.
# The quality department owns this number; it is one constant, here.
SEVERSTAL_TIER: str = "medium"

# Tier used for a class the knowledge base has never heard of -- a head this
# module was not built for. Scoring it as the middle tier keeps it visible in
# the report; scoring it 0 would hide a real box behind a clean-looking number,
# which on an inspection line is the worse failure. See `score_detection`.
UNKNOWN_CLASS_TIER: str = "medium"

# Root cause / base severity / recommended action, written for a line operator on a
# hot strip mill. "severity" here is the *base tier* of the defect type; the
# per-detection severity reported at runtime scales this by confidence and size.
DEFECT_INFO: dict[str, dict[str, str]] = {
    "crazing": {
        "cause": (
            "Network of fine surface cracks from low hot ductility at the strip "
            "surface: over-soaking or excessive reheat temperature coarsening the "
            "grain, finishing or coiling temperature off schedule, or tramp copper "
            "and tin from the scrap charge causing hot shortness (Cu above roughly "
            "0.2 wt% without compensating nickel)."
        ),
        "severity": "high",
        "action": (
            "Check reheat furnace soak temperature and residence time against the "
            "practice sheet, verify finishing and coiling temperatures on the "
            "pyrometer trend, pull the heat chemistry for Cu/Sn residuals, and hold "
            "the coil for surface conditioning or grinding before it is released."
        ),
    },
    "inclusion": {
        "cause": (
            "Non-metallic inclusion (oxide, silicate or sulphide) trapped below the "
            "slab surface and exposed and elongated by rolling. Sources are slag or "
            "mould-flux carryover in the caster, deoxidation products that failed to "
            "float out, and eroded ladle or tundish refractory."
        ),
        "severity": "critical",
        "action": (
            "Trace the coil back to its heat and cast sequence and quarantine the "
            "affected strand length. Review tundish level control and slag carryover "
            "at ladle change, inspect ladle and shroud refractory wear, and check "
            "calcium treatment and argon stirring practice. Crop or downgrade the "
            "affected length; inclusions cannot be rolled or pickled out."
        ),
    },
    "patches": {
        "cause": (
            "Irregular light or dark areas from non-uniform surface condition: "
            "patchy descaling that leaves residual scale which is then flattened by "
            "the rolls, temperature streaks from unbalanced reheat furnace burners, "
            "or uneven roll-coolant and emulsion pickup on the strip."
        ),
        "severity": "medium",
        "action": (
            "Verify descaler header pressure and check for blocked or misaligned "
            "nozzles across the strip width, balance reheat furnace burners to remove "
            "skid and burner temperature streaks, and confirm pickling line acid "
            "concentration and line speed so the patches are not carried forward."
        ),
    },
    "pitted_surface": {
        "cause": (
            "Dense field of small depressions left where scale nodules grew into the "
            "base metal and then spalled off, driven by prolonged high-temperature "
            "oxidation in the reheat furnace with an oxidising atmosphere, or by "
            "over-pickling and acid attack on the pickle line."
        ),
        "severity": "high",
        "action": (
            "Reduce furnace residence time and trim the air/fuel ratio towards "
            "stoichiometric to cut secondary scale growth, confirm descaling is "
            "effective ahead of the finishing stands, and audit pickle line acid "
            "concentration, bath temperature and line speed. Reject for exposed or "
            "coated applications: pits retain acid and moisture."
        ),
    },
    "rolled-in_scale": {
        "cause": (
            "Iron oxide scale pressed into the strip surface by the work rolls. "
            "Caused by incomplete primary or secondary descaling (low header "
            "pressure, blocked nozzles, wrong standoff), heavy furnace scaling, or "
            "scale build-up baked onto the work roll barrel."
        ),
        "severity": "high",
        "action": (
            "Restore descaler pressure to practice (typically above 180 bar) and "
            "clear or replace blocked nozzles, inspect work roll surface condition "
            "and shorten the roll change interval, and lower reheat temperature or "
            "residence time. Flag the coil: embedded oxide breaks out in cold "
            "rolling and ruins coating adhesion, and pickling may not fully remove it."
        ),
    },
    "scratches": {
        "cause": (
            "Linear mechanical gouging from contact with fixed or damaged plant: "
            "seized or non-rotating table rollers, worn side guides and side guards, "
            "damaged pinch rolls or coiler mandrel, hard scale debris dragged along "
            "the strip, or mishandling during coil transport and strapping."
        ),
        "severity": "medium",
        "action": (
            "Walk the run-out table and inspect for seized or scored rollers, dress "
            "or replace damaged side guides, guards and pinch rolls, clear scale "
            "debris from the table, and review coil handling, tong and strapping "
            "practice. Deep scratches act as crack initiators in forming and must be "
            "ground out or the length cropped."
        ),
    },
}

# The four Severstal classes of the joint head. Deliberately written as an
# admission rather than as metallurgy: see the SEVERSTAL_TIER note above for what
# the dataset does and does not publish. The wording is neutral on purpose so a
# reader cannot mistake it for a root-cause diagnosis, and it names the one
# action that is actually correct today -- get a human to classify it.
for _sev_index, _sev_name in enumerate(SEVERSTAL_CLASS_NAMES, start=1):
    DEFECT_INFO[_sev_name] = {
        "cause": (
            f"Unnamed surface defect, Severstal class {_sev_index}, learned from "
            "hot-rolled carbon steel strip on a different line. The source release "
            "publishes only the numeric class id: no defect name, no metallurgical "
            "description and no grading rule, so no root cause is asserted here. "
            "What the model has learned is that this texture is not clean steel "
            "and is not one of the six NEU-DET families -- that is the whole claim."
        ),
        "severity": SEVERSTAL_TIER,
        "action": (
            "Treat as an unclassified defect: pull the frame for visual grading "
            "before the coil is dispositioned, and check whether it repeats at a "
            "fixed position across the strip (plant contact) or wanders (process). "
            f"The quality department must assign class {_sev_index} a real tier and "
            "action from mill experience before it drives a release decision. The "
            f"tier on this entry is the neutral placeholder SEVERSTAL_TIER = "
            f"'{SEVERSTAL_TIER}', not a judgement about this defect."
        ),
    }
del _sev_index, _sev_name


# Reporting tiers, least to most serious, and the 0-100 score cutoffs between
# them. Public because the UI and the coil report band their own frame-level
# scores: a band must mean the same thing in every artefact this system emits.
SEVERITY_BANDS: tuple[str, ...] = ("low", "medium", "high", "critical")
_SEVERITY_CUTOFFS: tuple[float, ...] = (30.0, 55.0, 80.0)

# Score anchors for each base tier, on the same 0-100 scale as severity_score.
_TIER_BASE_SCORE: dict[str, float] = dict(zip(SEVERITY_BANDS, (22.0, 42.0, 64.0, 84.0)))

# A detection covering this fraction of the frame is treated as fully grown; above
# it the area multiplier saturates.
_AREA_SATURATION = 0.25
_AREA_GAIN = 0.55
# Geometric damping applied to the 2nd, 3rd, ... worst detection when aggregating
# to a single strip-quality risk score.
_AGGREGATION_DECAY = 0.6


def _severity_bucket(score: float) -> str:
    """Map a 0-100 defect score onto the four reporting tiers."""
    for cutoff, band in zip(_SEVERITY_CUTOFFS, SEVERITY_BANDS):
        if score < cutoff:
            return band
    return SEVERITY_BANDS[-1]


def score_detection(class_name: str, confidence: float, area_frac: float) -> float:
    """Score one detection on 0-100.

    The rule, kept deliberately in one place so it can be tuned with the quality
    department:

        base       = tier anchor for the defect class (DEFECT_INFO[...]["severity"])
        conf_mult  = 0.5 + 0.5 * confidence      -> a marginal hit is halved
        area_mult  = 1 + 0.55 * min(1, area_frac / 0.25)
                                                 -> a defect covering a quarter of
                                                    the frame is fully weighted
        score      = clip(base * conf_mult * area_mult, 0, 100)

    Confidence can only reduce the tier anchor, area can only raise it, so a
    high-confidence full-width inclusion saturates at 100 while a faint speck of the
    same class lands in the low band.

    A class the knowledge base does not carry is scored at `UNKNOWN_CLASS_TIER`
    rather than raising. This used to be a `KeyError` and it was a live ship
    blocker: the 10-class joint checkpoint loads fine, emits `severstal_*` boxes,
    and every one of them killed the call inside `_finalise` --

        KeyError: 'class_8'   (src/inference.py, via predict / predict_batch)

    -- which is why that checkpoint could not be served despite scoring better.
    All ten classes of the shipped heads are now in DEFECT_INFO, so this fallback
    should never fire; it exists so an unrecognised head degrades to a scored,
    visible detection instead of taking the inspection station down.
    """
    info = DEFECT_INFO.get(class_name)
    base = _TIER_BASE_SCORE[info["severity"] if info else UNKNOWN_CLASS_TIER]
    conf_mult = 0.5 + 0.5 * float(np.clip(confidence, 0.0, 1.0))
    area_mult = 1.0 + _AREA_GAIN * min(1.0, max(0.0, area_frac) / _AREA_SATURATION)
    return float(np.clip(base * conf_mult * area_mult, 0.0, 100.0))


def aggregate_severity(scores: Sequence[float]) -> float:
    """Combine per-detection scores into one strip-quality risk score on 0-100.

    A damped noisy-OR: the worst defect sets the floor, and each further defect
    closes a geometrically shrinking share of the remaining headroom. This keeps the
    score monotone in both defect count and defect severity, bounded at 100, and
    equal to the single detection's own score when only one is present. Returns 0.0
    for a clean frame.
    """
    if not scores:
        return 0.0
    residual = 1.0
    for rank, score in enumerate(sorted(scores, reverse=True)):
        residual *= 1.0 - (score / 100.0) * (_AGGREGATION_DECAY**rank)
    return float(np.clip(100.0 * (1.0 - residual), 0.0, 100.0))


@dataclass
class Detection:
    """One defect instance located in original-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    area_px: float
    area_frac: float
    severity: str
    # The 0-100 score `severity` is the band of. Carried on the record so the
    # frame-level aggregate is computed from the same numbers the UI displays,
    # and so aggregation never has to re-look-up a class name it may not know.
    # Defaults to 0.0 for hand-built records in tests and callers.
    severity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "area_px": round(float(self.area_px), 2),
            "area_frac": round(float(self.area_frac), 6),
            "severity": self.severity,
            "severity_score": round(float(self.severity_score), 2),
        }


@dataclass
class InferenceResult:
    """Everything the UI and the reporting layer need for a single frame."""

    detections: list[Detection] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    model_name: str = ""
    conf_threshold: float = 0.0
    iou_threshold: float = 0.0
    verdict: str = "PASS"
    dominant_class: str | None = None
    max_confidence: float = 0.0
    defect_count: int = 0
    severity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "defect_count": int(self.defect_count),
            "dominant_class": self.dominant_class,
            "max_confidence": round(float(self.max_confidence), 4),
            "severity_score": round(float(self.severity_score), 2),
            "detections": [d.to_dict() for d in self.detections],
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "timing_ms": {
                "preprocess": round(float(self.preprocess_ms), 3),
                "inference": round(float(self.inference_ms), 3),
                "postprocess": round(float(self.postprocess_ms), 3),
                "total": round(float(self.total_ms), 3),
            },
            "model_name": self.model_name,
            "conf_threshold": round(float(self.conf_threshold), 4),
            "iou_threshold": round(float(self.iou_threshold), 4),
        }


def _finalise(
    detections: list[Detection],
    *,
    image_size: tuple[int, int],
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    total_ms: float,
    model_name: str,
    conf: float,
    iou: float,
) -> InferenceResult:
    """Derive the frame-level verdict fields from a finished detection list."""
    dominant = max(detections, key=lambda d: d.confidence) if detections else None
    return InferenceResult(
        detections=detections,
        image_size=image_size,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
        total_ms=total_ms,
        model_name=model_name,
        conf_threshold=conf,
        iou_threshold=iou,
        verdict="DEFECT" if detections else "PASS",
        dominant_class=dominant.class_name if dominant else None,
        max_confidence=float(dominant.confidence) if dominant else 0.0,
        defect_count=len(detections),
        # From the scores already on the records: one lookup per detection, in
        # `_build_detections`, and no second chance to disagree with the UI.
        severity_score=aggregate_severity([d.severity_score for d in detections]),
    )


# The 10-class joint checkpoint. Not the default (see `resolve_weights`), but
# named here so calling it is one import rather than a path literal copied into
# five scripts:
#
#     detector = load_detector(weights=inference.JOINT_WEIGHTS)
JOINT_WEIGHTS = MODELS_DIR / "yolov8n_joint" / "weights" / "best.pt"

# Checkpoint preference, best first, as data rather than a literal inside the
# function. `yolov8n_joint` sits below both NEU-DET runs deliberately -- it is
# the better detector (see `resolve_weights`) but it is not yet the one this
# tree is wired for. It is listed at all so that a tree with the NEU-DET runs
# deleted serves the joint model rather than falling through to an mtime glob.
_PREFERENCE_ORDER: tuple[str, ...] = (
    "yolov8n_neudet/weights/best.pt",
    "yolov8n_neudet/weights/last.pt",
    "yolov8s_neudet/weights/best.pt",
    "yolov8s_neudet/weights/last.pt",
    "yolov8n_joint/weights/best.pt",
)


def resolve_weights(preferred: str | Path | None = None) -> Path:
    """Locate the checkpoint to serve, newest-run-wins as the final fallback.

    The nano run is preferred over the small one, which is not the usual ordering
    and is not a typo. Both were fine-tuned at imgsz 320 for up to 150 epochs (nano
    early-stopped at 135, small ran all 150). At 320 px on the held-out test split
    nano wins on every headline metric -- mAP50 0.7286 against 0.6598, mAP50-95
    0.3926 against 0.3618, and all six per-class AP50 values -- and the paired
    bootstrap in `reports/model_study.json` puts the mAP50 gap at +0.069 with a 95%
    interval of [+0.031, +0.099], which excludes zero.

    The honest qualifier, which the earlier version of this docstring omitted: give
    each model its own best input size (256 px, chosen on val) and the gap shrinks
    to +0.018 mAP50 with a 95% interval of [-0.008, +0.042], which *contains* zero,
    and small is marginally ahead on mAP50-95. So the claim behind this ordering is
    not "nano is the better detector"; it is "nano is not worse, and it is 3.7x
    smaller, 3.5x cheaper in FLOPs and roughly twice as fast". That is enough to
    decide which checkpoint the runtime serves by default.

    SHOULD THIS PREFER models/yolov8n_joint INSTEAD? RECOMMENDATION: YES, ONCE
    THE FOUR ITEMS BELOW ARE DONE -- AND NOT BEFORE. It is not flipped here.

    The case FOR the joint checkpoint, re-measured on this working tree rather
    than quoted from a report:

    * In domain it is not worse. On the same 180 held-out NEU-DET test frames
      and the same 446 instances, at imgsz 256 on cpu:
          yolov8n_neudet  mAP50 0.7524  mAP50-95 0.3967  P 0.6960  R 0.6867
          yolov8n_joint   mAP50 0.7642  mAP50-95 0.4008  P 0.6946  R 0.7055
      (`YOLO(w).val(data=..., imgsz=256, device="cpu")`, the joint run through
      `data/joint_xdsafe/eval_neu_test.yaml` because a 10-class head cannot be
      scored against a 6-class yaml.) +0.0118 mAP50 is inside the noise: the
      paired bootstrap in `reports/model_study.json` put a *larger* gap, +0.018,
      at 95% [-0.008, +0.042], which contains zero. So read this as a tie, not
      as a win.
    * Out of domain it is a different class of model. On verified defect-free
      Severstal strip at the shipped operating point (`reports/gap1_cross_domain.json`),
      clean-frame false alarms fall 93.7% -> 32.5%, boxes on a clean tile fall
      2.02 -> 0.070, and defect-vs-clean AUC on that line rises 0.608
      [0.576, 0.640] -> 0.958 [0.948, 0.968]. Cross-domain recall rises at the
      same time (0.720 -> 0.866), so this is not the threshold being tightened.
      A detector that alarms on 94% of clean coils cannot be put in front of an
      operator; that is the argument, and it is decisive.

    The case AGAINST, which is why the switch is not made in this commit:

    * Its in-domain crop-level separation is slightly WORSE, not better: NEU-DET
      defect vs mined-clean AUC 0.968 [0.948, 0.984] -> 0.940 [0.905, 0.970],
      scale-stratified 0.960 -> 0.905. The intervals overlap, but the point
      estimate moves the wrong way and no one has explained why.
    * Four of its ten classes have no agreed meaning and a placeholder tier
      (SEVERSTAL_TIER). Serving it by default puts "severstal_3, medium" in
      front of a mill operator as if the system knew what that was.
    * Three consumers still assume a six-class head, and all three default to
      this function, so flipping it here regresses them silently:
        - `demo/app.py:preferred_checkpoint_index` hard-codes `yolov8n_neudet`,
          so the console would open on one checkpoint while the report generator
          served another. `tests/test_smoke.py` asserts these two agree, on
          purpose.
        - `src/false_alarm.py` (~line 892) counts false positives with
          `if 0 <= cls < len(CLASS_NAMES)`, so `severstal_*` boxes vanish from
          the class table while still counting in the total.
        - `src/export_model.py` (~line 389) does `CLASS_NAMES[int(ref[0, 5])]`,
          which is an IndexError the moment the top box is class 6-9.
      Also `reports/operating_point.json` and the calibration curve in
      `reports/calibration.json` were both fitted on the shipped checkpoint.

    TO FLIP IT, in one coordinated change: (1) give the four Severstal classes a
    real tier and action from the quality department, (2) fix the two
    `len(CLASS_NAMES)` assumptions above to use the detector's own
    `class_names`, (3) move `yolov8n_joint` to the front of `_PREFERENCE_ORDER`
    and update `demo/app.py:preferred_checkpoint_index` in the same commit, and
    (4) re-run `src/calibrate.py` and the operating-point selection on the joint
    checkpoint. Until then the previous behaviour is the behaviour, and the
    joint checkpoint is one keyword away: `load_detector(weights=JOINT_WEIGHTS)`.
    """
    if preferred is not None:
        path = Path(preferred).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Requested weights do not exist: {path}")
        return path

    for relative in _PREFERENCE_ORDER:
        candidate = MODELS_DIR / relative
        if candidate.is_file():
            return candidate

    fallbacks = sorted(
        MODELS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if fallbacks:
        return fallbacks[0]

    raise FileNotFoundError(
        f"No detector checkpoint found under {MODELS_DIR}. Train one first with:\n"
        f"  {PROJECT_ROOT / '.venv/bin/python'} {PROJECT_ROOT / 'src/train_detector.py'} "
        f"--model yolov8n.pt --name yolov8n_neudet\n"
        "or pass an explicit path, e.g. load_detector(weights='/path/to/best.pt')."
    )


def resolve_device(device: str = "auto") -> str:
    """auto -> mps -> cuda -> cpu. Any explicit string is passed through."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _synchronise(device: str) -> None:
    """Block until queued accelerator work has finished, so timings are honest."""
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda") or device.isdigit():
        torch.cuda.synchronize()


# Ultralytics times its own preprocess/inference/postprocess stages with
# `utils.ops.Profile`, which synchronises only for cuda/npu/xpu. On MPS the
# inference stage therefore stops its clock while the GPU is still working, and
# the cost reappears in the postprocess stage -- the first place a tensor is
# pulled back to the host. Measured here at 320 px, batch 1: the unsynchronised
# split reads 3.5 ms inference / 13.6 ms postprocess, which invites the wrong
# conclusion that the pipeline is NMS-bound; with the stage clocks synchronised
# the same call splits 15.7 ms inference / 1.5 ms postprocess. Wall-clock totals
# are unaffected either way, because `_forward` already synchronises at the end.
#
# The two extra synchronisations cost roughly 2% of end-to-end latency. Set
# STAGE_TIMING_SYNC = False before constructing a detector to buy that back and
# accept a meaningless stage split.
STAGE_TIMING_SYNC = True
_PROFILER_PATCHED = False


def _install_synchronised_profiler(device: str) -> bool:
    """Make the framework's stage clocks accelerator-aware. Idempotent, best effort.

    Returns True if per-stage timings on `device` can be trusted. A failure here
    is not fatal: it costs stage attribution, not correctness, so it degrades to
    the framework default rather than refusing to run.
    """
    global _PROFILER_PATCHED
    if not device.startswith("mps"):
        return True  # cuda/xpu already synchronise; cpu is synchronous by nature
    if not STAGE_TIMING_SYNC:
        return False
    if _PROFILER_PATCHED:
        return True
    try:
        from ultralytics.utils import ops as _ops

        base = _ops.Profile

        class _SynchronisedProfile(base):  # type: ignore[misc, valid-type]
            """Profile whose clock waits for queued MPS work before it reads."""

            def time(self) -> float:
                torch.mps.synchronize()
                return time.perf_counter()

        # The predictor builds its profiler tuple per call from this attribute,
        # so replacing it here reaches every subsequent predict().
        _ops.Profile = _SynchronisedProfile
        _PROFILER_PATCHED = True
        return True
    except Exception:  # noqa: BLE001 - degrade to framework timing, never crash
        return False


def _to_rgb(image: str | Path | np.ndarray | Any) -> np.ndarray:
    """Normalise any accepted input to a contiguous HWC RGB uint8 array."""
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB array, got shape {image.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    # PIL.Image without importing PIL at module scope.
    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))

    raise TypeError(
        f"Unsupported image type {type(image)!r}; expected path, PIL.Image or ndarray."
    )


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Ultralytics interprets bare ndarray sources as BGR, so convert explicitly."""
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _head_class_names(names: Any) -> list[str] | None:
    """The checkpoint's own class list, or None if it does not carry a usable one.

    Ultralytics stores names as `{index: name}`. A list is only returned when the
    keys are exactly 0..n-1 and every value is a non-empty string, so a partial or
    sparse mapping falls back to the module contract rather than producing a list
    with holes in it that `_build_detections` would index into.
    """
    if not names:
        return None
    try:
        items = {int(k): v for k, v in dict(names).items()}
    except (TypeError, ValueError):
        return None
    if set(items) != set(range(len(items))):
        return None
    if not all(isinstance(v, str) and v.strip() for v in items.values()):
        return None
    return [items[i] for i in range(len(items))]


class DefectDetector:
    """Thin, opinionated wrapper around the trained YOLO detector."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from ultralytics import YOLO  # deferred: keeps module import free of side effects

        self.weights_path = resolve_weights(weights)
        self.device = resolve_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)

        self.stage_timing_synchronised = _install_synchronised_profiler(self.device)
        self._model = YOLO(str(self.weights_path))
        self._model.to(self.device)
        # The class list comes off the checkpoint that was actually loaded,
        # whatever width its head is. The previous rule was
        # `if len(names) == len(CLASS_NAMES)`, i.e. "trust the head only if it
        # has exactly six classes", which silently mislabelled every 10-class
        # checkpoint: indices 6-9 fell off the end of the six-name list and were
        # reported as `class_6` .. `class_9`, with no colour and no knowledge
        # base entry. Falling back to CLASS_NAMES is now only for a checkpoint
        # that carries no usable names at all.
        self.class_names = _head_class_names(getattr(self._model, "names", None))
        if self.class_names is None:
            self.class_names = list(CLASS_NAMES)

    @property
    def model_name(self) -> str:
        run_dir = self.weights_path.parent.parent.name
        return f"{run_dir}/{self.weights_path.name}" if run_dir else self.weights_path.name

    def __repr__(self) -> str:
        return (
            f"DefectDetector(model={self.model_name!r}, device={self.device!r}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz})"
        )

    # ---------------------------------------------------------------- internals

    def _forward(self, bgr_batch: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
        """Run the network on BGR arrays; return per-image [x1,y1,x2,y2,conf,cls]."""
        results = self._model.predict(
            bgr_batch,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        _synchronise(self.device)
        raw = [r.boxes.data.detach().cpu().numpy().astype(np.float64) for r in results]
        speed = results[0].speed if results else {}
        timing = {
            "preprocess": float(speed.get("preprocess", 0.0)),
            "inference": float(speed.get("inference", 0.0)),
            "postprocess": float(speed.get("postprocess", 0.0)),
        }
        return raw, timing

    def _build_detections(self, raw: np.ndarray, width: int, height: int) -> list[Detection]:
        """Turn a raw (N,6) box array into scored, named Detection records."""
        frame_area = float(max(width * height, 1))
        detections: list[Detection] = []
        for x1, y1, x2, y2, conf, cls in raw:
            cls_id = int(cls)
            name = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            # Clamp into the frame, far corner never ahead of the near one. A box
            # the head placed wholly outside (letterbox rounding on an extreme
            # aspect ratio does produce these) collapses to a zero-area box at the
            # edge rather than to an inverted one, so `x1 <= x2` holds for every
            # Detection the UI, the report and the exporters ever see.
            x1c = min(max(0.0, float(x1)), float(width))
            y1c = min(max(0.0, float(y1)), float(height))
            x2c = min(max(x1c, float(x2)), float(width))
            y2c = min(max(y1c, float(y2)), float(height))
            area = (x2c - x1c) * (y2c - y1c)
            area_frac = area / frame_area
            score = score_detection(name, float(conf), area_frac)
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=name,
                    confidence=float(conf),
                    bbox_xyxy=(x1c, y1c, x2c, y2c),
                    area_px=area,
                    area_frac=area_frac,
                    severity=_severity_bucket(score),
                    severity_score=score,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------- public

    def warmup(self, n: int = 3) -> None:
        """Pay the lazy kernel-compilation cost up front so the first real frame
        is not an outlier. Cheap: n forward passes on a synthetic grey frame."""
        blank = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        for _ in range(max(0, int(n))):
            self._forward([blank])

    def predict(self, image: str | Path | np.ndarray | Any) -> InferenceResult:
        """Detect defects in a single frame.

        Latency accounting: `total_ms` is the true wall clock for this call.
        `preprocess_ms` covers decode/colour conversion here plus the framework's
        letterbox and host-to-device copy, `inference_ms` is the forward pass, and
        `postprocess_ms` is NMS plus building the scored records. The three stages
        are measured independently, so their sum is slightly below `total_ms`; the
        residual is Python and framework dispatch overhead.
        """
        t0 = time.perf_counter()
        rgb = _to_rgb(image)
        height, width = rgb.shape[:2]
        bgr = _rgb_to_bgr(rgb)
        t1 = time.perf_counter()

        raw, timing = self._forward([bgr])
        t2 = time.perf_counter()

        detections = self._build_detections(raw[0], width, height)
        t3 = time.perf_counter()

        return _finalise(
            detections,
            image_size=(width, height),
            preprocess_ms=(t1 - t0) * 1000.0 + timing["preprocess"],
            inference_ms=timing["inference"],
            postprocess_ms=timing["postprocess"] + (t3 - t2) * 1000.0,
            total_ms=(t3 - t0) * 1000.0,
            model_name=self.model_name,
            conf=self.conf,
            iou=self.iou,
        )

    def predict_batch(
        self, images: Sequence[Any] | Iterable[Any], batch_size: int = 16
    ) -> list[InferenceResult]:
        """Score many frames, chunked so device memory stays bounded.

        Per-image timings inside a chunk are the chunk's mean: the framework times a
        batched forward pass as a whole, and splitting it per image would be a
        fabrication. `total_ms` is the chunk wall clock divided by chunk size, which
        is the number that matters for throughput planning.
        """
        items = list(images)
        if not items:
            return []
        chunk = max(1, int(batch_size))
        out: list[InferenceResult] = []

        for start in range(0, len(items), chunk):
            group = items[start : start + chunk]
            t0 = time.perf_counter()
            rgbs = [_to_rgb(img) for img in group]
            bgrs = [_rgb_to_bgr(rgb) for rgb in rgbs]
            t1 = time.perf_counter()

            raw, timing = self._forward(bgrs)
            t2 = time.perf_counter()

            per_image_pre = (t1 - t0) * 1000.0 / len(group)
            built: list[list[Detection]] = []
            for rgb, raw_i in zip(rgbs, raw):
                h, w = rgb.shape[:2]
                built.append(self._build_detections(raw_i, w, h))
            t3 = time.perf_counter()

            per_image_post = (t3 - t2) * 1000.0 / len(group)
            per_image_total = (t3 - t0) * 1000.0 / len(group)
            for rgb, detections in zip(rgbs, built):
                h, w = rgb.shape[:2]
                out.append(
                    _finalise(
                        detections,
                        image_size=(w, h),
                        preprocess_ms=per_image_pre + timing["preprocess"],
                        inference_ms=timing["inference"],
                        postprocess_ms=timing["postprocess"] + per_image_post,
                        total_ms=per_image_total,
                        model_name=self.model_name,
                        conf=self.conf,
                        iou=self.iou,
                    )
                )
        return out

    def predict_tiled(
        self,
        image: str | Path | np.ndarray | Any,
        tile: int | None = None,
        overlap: float = 0.2,
    ) -> InferenceResult:
        """Sliding-window inference for a full-width strip image.

        A 2 m strip scanned at line resolution is thousands of pixels wide; squashing
        it to one network input destroys the fine texture that separates crazing from
        pitting.
        Instead the frame is cut into overlapping tiles, each tile is run at native
        scale, boxes are offset back into full-image coordinates, and a single
        class-aware NMS is applied across every tile so a defect straddling a seam is
        merged rather than counted twice.

        Images that already fit inside one tile fall through to `predict()`.

        LOSSLESS CONTRACT, at the default `tile is None` (i.e. tile == imgsz).
        Verified on a real 2048x1000 GC10 line-scan frame by spying on `cv2.resize`
        and on `ultralytics.data.augment.LetterBox.__call__` for the whole call:

        * 0 calls to `cv2.resize` in the entire tiled pass (50 tiles),
        * all 50 LetterBox invocations returned an array byte-identical to their
          input, every one 256x256x3 in and 256x256x3 out -- the framework's
          letterbox short-circuits when `r == 1.0` and the padding is zero, so
          the tensor that reaches the network IS the crop,
        * all 50 crops compared equal under `np.array_equal` to the source
          sub-rectangle `rgb[y0:y0+h, x0:x0+w]`,
        * every source pixel is seen at least once (1 to 4 times at overlap 0.2).

        For contrast, `predict()` on the same frame makes exactly one resize,
        (1000, 2048) -> (256, 125): 32,000 of the frame's 2,048,000 pixels reach
        the network, 1.56%, at a linear scale of 0.125. Measured detections on
        the three labelled GC10 strips in `assets/`, class-agnostic, at the
        shipped conf 0.15: whole-frame downscale localises 1 of the 6 labelled
        defects, tiling localises 5 of 6.

        THE CONTRACT HOLDS ONLY AT tile == imgsz, and that is why the default is
        not a round number someone liked. Measured the same way on the same
        frame with imgsz 256: `tile=128` produced 128 resize calls, every one a
        (128,128) -> (256,256) UPscale that invents pixels; `tile=512` produced 8
        calls, every one a (512,512) -> (256,256) DOWNscale that throws three
        quarters of them away. A non-default tile is a legitimate speed/coverage
        trade, but it is resampling and it is not what this method advertises.

        The other resampling path is the fall-through above: a frame that fits in
        one tile goes to `predict()` and is letterboxed like any other frame -- a
        200x200 NEU-DET crop at imgsz 256 is a 1.28x upscale. Tiling cannot avoid
        that, because there is nothing to tile.
        """
        # A tile larger than the network input is not "native scale": the crop is
        # cut at `tile` and then letterboxed down to `self.imgsz` before the network
        # sees it, so tile > imgsz is downscaling with extra steps. Defaulting the
        # tile to the input size is the only setting that keeps one source pixel on
        # one network pixel, which is the whole reason this method exists.
        tile = self.imgsz if tile is None else int(tile)

        t0 = time.perf_counter()
        rgb = _to_rgb(image)
        height, width = rgb.shape[:2]
        if width <= tile and height <= tile:
            # Charge the decode we already paid to the single-pass result rather
            # than silently dropping it from the reported latency.
            decode_ms = (time.perf_counter() - t0) * 1000.0
            result = self.predict(rgb)
            result.preprocess_ms += decode_ms
            result.total_ms += decode_ms
            return result

        # Above ~0.9 the stride collapses and the tile count explodes; clamp rather
        # than let a bad UI slider stall the line.
        overlap = float(np.clip(overlap, 0.0, 0.9))
        tile_w, tile_h = min(int(tile), width), min(int(tile), height)
        stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
        stride_y = max(1, int(round(tile_h * (1.0 - overlap))))
        xs = _tile_origins(width, tile_w, stride_x)
        ys = _tile_origins(height, tile_h, stride_y)

        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for y0 in ys:
            for x0 in xs:
                crops.append(_rgb_to_bgr(rgb[y0 : y0 + tile_h, x0 : x0 + tile_w]))
                offsets.append((x0, y0))
        t1 = time.perf_counter()

        # Chunked so a very wide strip cannot blow up device memory in one go.
        raw_tiles: list[np.ndarray] = []
        inference_ms = 0.0
        fw_pre_ms = 0.0
        fw_post_ms = 0.0
        chunk = 8
        for start in range(0, len(crops), chunk):
            group = crops[start : start + chunk]
            raw, timing = self._forward(group)
            raw_tiles.extend(raw)
            n = len(group)
            fw_pre_ms += timing["preprocess"] * n
            inference_ms += timing["inference"] * n
            fw_post_ms += timing["postprocess"] * n
        t2 = time.perf_counter()

        boxes: list[np.ndarray] = []
        for (x0, y0), raw in zip(offsets, raw_tiles):
            if raw.size == 0:
                continue
            shifted = raw.copy()
            shifted[:, [0, 2]] += x0
            shifted[:, [1, 3]] += y0
            boxes.append(shifted)

        if boxes:
            merged = np.concatenate(boxes, axis=0)
            keep = torchvision.ops.batched_nms(
                torch.from_numpy(merged[:, :4]).float(),
                torch.from_numpy(merged[:, 4]).float(),
                torch.from_numpy(merged[:, 5]).long(),
                self.iou,
            ).numpy()
            merged = merged[keep]
        else:
            merged = np.zeros((0, 6), dtype=np.float64)

        detections = self._build_detections(merged, width, height)
        t3 = time.perf_counter()

        return _finalise(
            detections,
            image_size=(width, height),
            preprocess_ms=(t1 - t0) * 1000.0 + fw_pre_ms,
            inference_ms=inference_ms,
            postprocess_ms=fw_post_ms + (t3 - t2) * 1000.0,
            total_ms=(t3 - t0) * 1000.0,
            model_name=self.model_name,
            conf=self.conf,
            iou=self.iou,
        )

    def annotate(
        self,
        image: str | Path | np.ndarray | Any,
        result: InferenceResult,
        show_conf: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw `result` over `image` and return a new HWC RGB uint8 array.

        Geometry scales with the frame so the overlay reads the same on a 200 px
        NEU-DET crop and on a 4000 px strip capture. Labels sit on a filled chip in
        the class colour with automatically contrasting text, because thin outlined
        text disappears against light grey steel.
        """
        canvas = _to_rgb(image).copy()
        height, width = canvas.shape[:2]
        ref = max(height, width)

        thickness = max(1, int(round(ref / 400.0)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(np.clip(ref / 1200.0, 0.34, 2.0))
        font_thickness = max(1, int(round(font_scale * 1.6)))
        pad = max(2, int(round(ref / 250.0)))

        for det in result.detections:
            colour = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

            if not show_labels:
                continue
            label = det.class_name
            if show_conf:
                label = f"{label} {det.confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            chip_w = text_w + 2 * pad
            chip_h = text_h + baseline + 2 * pad
            chip_x = min(max(0, x1), max(0, width - chip_w))
            # Prefer above the box; drop inside the box when there is no room.
            chip_y = y1 - chip_h if y1 - chip_h >= 0 else min(y1, height - chip_h)
            chip_y = max(0, chip_y)

            cv2.rectangle(
                canvas,
                (chip_x, chip_y),
                (chip_x + chip_w, chip_y + chip_h),
                colour,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (chip_x + pad, chip_y + chip_h - baseline - pad),
                font,
                font_scale,
                _contrast_colour(colour),
                font_thickness,
                cv2.LINE_AA,
            )
        return canvas


def _tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """Sliding-window origins covering `extent`, last window flush to the edge."""
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def _contrast_colour(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white text, whichever is legible on this chip (Rec. 709 luma)."""
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (20, 20, 20) if luma > 140.0 else (255, 255, 255)


_DETECTOR_CACHE: dict[tuple, DefectDetector] = {}
_CACHE_LOCK = threading.Lock()


def load_detector(
    weights: str | Path | None = None, device: str = "auto", **kw: Any
) -> DefectDetector:
    """Process-level cached detector, keyed on the full construction signature.

    Streamlit reruns the whole script on every widget change; without this the
    checkpoint would be re-read from disk on each interaction.
    """
    conf = float(kw.pop("conf", 0.25))
    iou = float(kw.pop("iou", 0.45))
    imgsz = int(kw.pop("imgsz", DEFAULT_IMGSZ))
    if kw:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kw)}")

    key = (str(resolve_weights(weights)), resolve_device(device), conf, iou, imgsz)
    with _CACHE_LOCK:
        detector = _DETECTOR_CACHE.get(key)
        if detector is None:
            detector = DefectDetector(
                weights=key[0], device=key[1], conf=conf, iou=iou, imgsz=imgsz
            )
            _DETECTOR_CACHE[key] = detector
    return detector
