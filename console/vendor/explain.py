"""Visual explanations for the Jindal Stainless surface-defect detector.

An operator will not act on a red box unless the system can show *what the model
looked at*. This module produces two kinds of evidence, and is deliberately honest
about what each one is worth.

Why EigenCAM and not Grad-CAM
-----------------------------
Gradient CAMs (Grad-CAM, HiRes-CAM, Grad-CAM++) need a scalar score that is
differentiable back to the target layer's activations. That is not available on a
served ultralytics detector: setting up for inference clears `requires_grad` on
every parameter (measured: 0 of 184 parameters carry it after the first
`predict`), so the head's output carries no autograd graph. Asking
`pytorch_grad_cam.GradCAM` for a map on this model fails outright with

    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

(reproduced on the checkpoint in `models/`, layers 9 and 21, both GradCAM and
HiResCAM, with a real detection-head target). Conv+BatchNorm fusion is *not* the
cause here - `YOLO.is_fused()` is False and all 57 BatchNorm2d modules are still
present - the cleared gradient flags are. Re-enabling them would mutate the
process-wide cached detector that the UI and the report generator share. So this
module does not ship a gradient CAM at all.

`generate_cam` therefore uses **EigenCAM** (Muhammad & Yeasin, 2020): the first
principal component of the target layer's activation tensor. It needs a forward pass
only, so it explains exactly the fused network that is in production. Its honest
limitation is that it is **class-agnostic** - it shows where this layer's features
are strongest, not which class they voted for. It is read as "the model's attention",
not "the evidence for this specific box".

SVD has a sign ambiguity (`A = U S V^T = (-U) S (-V^T)`), and the CAM pipeline clips
negatives (`BaseCAM.compute_cam_per_layer` applies `np.maximum(cam, 0)` before it
scales), so an unflipped projection can return the exact complement of the salient
region. This is not hypothetical: on `inclusion_271.jpg` at layer 21, stock
`EigenCAM` returns a map with mean 0.305 and 26% of pixels above 0.5, while the
sign-corrected one has mean 0.065 and 1.4%. `_SignCorrectedEigenCAM` applies the
correction before the clip.

Why an occlusion fallback ships alongside it
--------------------------------------------
`occlusion_sensitivity` slides a grey patch over the frame and measures how far the
target detection's confidence falls. It is slow (one forward pass per patch position)
but it is model-agnostic, needs no gradients and no layer choice, and is *causal*:
a bright pixel means "covering this actually destroyed the detection". When a CAM and
an occlusion map disagree, believe the occlusion map.

Both paths are sanity-checked. A heatmap that is NaN, all-zero or perfectly uniform
is not an explanation, and this module raises `DegenerateExplanationError` rather than
handing a reviewer a picture that means nothing.

What this costs, and what that means for the UI
-----------------------------------------------
EigenCAM is *not* free, and the cost is almost entirely a one-off. It runs its own
instrumented forward pass with a hook on the target layer plus an SVD of the
activation tensor, none of which is shared with the detector's own `predict`.
Measured on this host (Apple M5, MPS, imgsz 256, `inclusion_271.jpg`):

    first CAM in a fresh process   1,236 ms in-process / 1,998 ms through the CLI
                                   (the project audit measured 2,505 ms)
    every CAM after that           11.5, 11.2, 11.0, 11.1, 10.9 ms
    the detection it explains      5.9 ms

The spread on the first call is MPS kernel compilation and the CAM object's own
construction, not the algorithm. Occlusion sensitivity has no such cliff and no
such floor: it is one forward pass per patch position, so it is slow *every* time.

A console must therefore render a CAM **on demand, behind an explicit button and a
spinner**, not inline on every frame and never inside a batch loop. The steady-state
11 ms would be affordable inline; the first click would not be, and the operator who
pays that stall is the one forming an opinion about whether the system is real time.
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/explain.py` from the Jindal Stainless surface-defect
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

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (  # noqa: E402  (path bootstrap must run first)
    DEFAULT_IMGSZ,
    DefectDetector,
    Detection,
    _to_rgb,  # deliberate: the CAM must see exactly the pixels the detector saw
    load_detector,
)

__all__ = [
    "DegenerateExplanationError",
    "Explanation",
    "DEFAULT_CAM_LAYER",
    "generate_cam",
    "overlay_cam",
    "occlusion_sensitivity",
    "explain",
    "heatmap_stats",
    "build_parser",
    "main",
]

# Index into `DetectionModel.model` (a 23-entry Sequential for YOLOv8).
# 21 is the deepest neck C2f, the P5/32 feature that feeds the Detect head directly.
#
# Measured on the VALIDATION split - never on test, which stays untouched so the
# reported detector metrics remain an honest held-out estimate. 55 val frames with
# detections (every third file), checkpoint yolov8s_neudet/best.pt at epoch 86/150,
# mAP50 0.687. Two criteria: `contrast` is the mean (CAM inside the predicted boxes
# minus CAM outside), which is what a reader intuitively wants but is biased by how
# much of the frame the boxes cover; `AUC` is the rank-equivalent, P(a random in-box
# pixel outranks a random outside pixel), which is not.
#
#     layer          contrast (+/- sem)     AUC
#      4             +0.061 +/- 0.008      0.603
#      6             +0.134 +/- 0.011      0.751
#      9  (SPPF)     +0.059 +/- 0.040      0.535
#     15  (P3 neck)  +0.075 +/- 0.008      0.683
#     18             +0.083 +/- 0.014      0.653
#     21             +0.096 +/- 0.011      0.695
#     [9, 21]        +0.080 +/- 0.026      0.607
#     [15, 18, 21]   +0.135 +/- 0.012      0.727
#
# So 21 does NOT top either criterion: layer 6 wins the AUC and [15, 18, 21] ties it
# on contrast. 21 is kept anyway, and the reason is not the separation score - it is
# that 6 buys its score with a blurrier map. On the six per-class test frames layer 6
# scores structure_ratio 9-17 against 19-22 for layer 21, spreads its mass over a
# wider region, and costs 130-180 ms per CAM against 70-120 ms, because the hook sits
# on a 4x larger feature map. A diffuse map that overlaps the box is worse evidence
# for an operator than a tight one that sometimes misses.
#
# This is a close call on a model that is still training. Re-measure both columns
# when the run finishes before treating 21 as settled; the gaps here are a few sem
# wide, not decisive.
DEFAULT_CAM_LAYER = 21

# Padding grey used for letterboxing and for the occluding patch. 114 is the
# ultralytics letterbox fill, so an occluded region looks to the network like the
# neutral border it already sees on every non-square frame.
_PAD_VALUE = 114

# Degeneracy thresholds, on the 0-1 normalised heatmap.
_MIN_STD = 0.01           # below this the map carries essentially no contrast
_MIN_DYNAMIC_RANGE = 0.05  # p99 - p01
# Ratio of measured block-to-block variation against what pure pixel noise of the
# same global std would produce. ~1.0 means the map is noise (measured: 0.75-1.12 on
# synthetic uniform and salt-and-pepper noise at 64x64 and 200x200); a localised map
# on a 200x200 NEU-DET frame measures 15-25. Treat this only as a floor, not as a
# quality score: the ratio grows with the upsampling factor from the CAM's native
# grid to the frame, so a 3000x3000 frame legitimately scores in the hundreds and
# the absolute value is not comparable across image sizes.
_MIN_STRUCTURE_RATIO = 2.0
_BLOCK_GRID = 8


class DegenerateExplanationError(RuntimeError):
    """Raised when a heatmap is not a usable explanation (NaN, flat or all-zero)."""


@dataclass(frozen=True)
class _Letterbox:
    """Geometry of a padded resize, kept so the heatmap can be mapped back."""

    scale: float
    left: int
    top: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int


@dataclass
class Explanation:
    """A heatmap plus the provenance a reviewer needs to judge it."""

    heatmap: np.ndarray = field(repr=False)
    method: str
    layer: str | None
    target: dict[str, Any] | None
    stats: dict[str, float]
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)

    def overlay(self, image: Any, alpha: float = 0.45) -> np.ndarray:
        """Convenience wrapper around `overlay_cam` for this explanation."""
        return overlay_cam(image, self.heatmap, alpha=alpha)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary. The heatmap array itself is not included."""
        return {
            "method": self.method,
            "layer": self.layer,
            "target": self.target,
            "stats": {k: round(float(v), 6) for k, v in self.stats.items()},
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------- geometry


def _letterbox(rgb: np.ndarray, size: int, stride: int = 32) -> tuple[np.ndarray, _Letterbox]:
    """Aspect-preserving resize + pad, replicating what the detector actually feeds.

    On a single frame the ultralytics predictor runs
    `LetterBox(check_imgsz(imgsz), auto=True, stride=model.stride)`, which pads only
    to the next multiple of `stride` - it does *not* pad out to a full `size` x
    `size` square. A 240x800 frame at imgsz 640 reaches the network as 192x640, not
    640x640. Padding to a square here would make the CAM explain a differently
    scaled, differently padded image from the one that drew the boxes: invisible on
    square NEU-DET crops, wrong on any wide strip capture. The rounding below (round
    on the resize, `round(pad/2 -/+ 0.1)` on the split) mirrors `LetterBox.get_params`
    exactly so the canvas is pixel-identical (verified against the real `LetterBox`
    across 44 size/imgsz combinations, including two that crash it outright).

    The one case this does not reproduce is a `predict_batch` call whose chunk holds
    *mixed* frame sizes: ultralytics then sets `auto=False` and pads every frame to a
    full square instead, so boxes from such a call live in a different geometry from
    this canvas. `report.inspect_coil` scores frames in same-size groups precisely so
    that case cannot arise; any other caller batching mixed sizes must do the same.
    """
    h, w = rgb.shape[:2]
    stride = max(1, int(stride))
    # ultralytics check_imgsz: the request is rounded up to a stride multiple.
    size = max(stride, int(np.ceil(float(size) / stride) * stride))

    scale = min(size / float(h), size / float(w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = (
        rgb
        if (new_w, new_h) == (w, h)
        else cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    )

    dw, dh = (-new_w) % stride, (-new_h) % stride
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * 3
    )
    return canvas, _Letterbox(scale, left, top, new_w, new_h, w, h)


def _unletterbox(canvas_map: np.ndarray, geom: _Letterbox) -> np.ndarray:
    """Crop the padding off a canvas-space map and restore original resolution."""
    cropped = canvas_map[
        geom.top : geom.top + geom.new_h, geom.left : geom.left + geom.new_w
    ]
    return cv2.resize(
        cropped, (geom.orig_w, geom.orig_h), interpolation=cv2.INTER_LINEAR
    )


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Scale a finite array to [0, 1]; a constant array becomes all-zero."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------- validation


def heatmap_stats(heatmap: np.ndarray) -> dict[str, float]:
    """Summary statistics used both for reporting and for degeneracy checks.

    `block_std` is the standard deviation across the means of an 8x8 grid of tiles,
    and `structure_ratio` divides it by the block-mean std that iid pixel noise of the
    same global std would produce (`std / sqrt(pixels_per_block)`). That normalisation
    matters: salt-and-pepper noise has a high global `std` *and* a non-trivial
    `block_std`, but a `structure_ratio` of about 1. Only a spatially organised map
    scores well above 1.
    """
    arr = np.asarray(heatmap, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if finite.size == 0:
        return {
            "min": float("nan"), "max": float("nan"), "mean": float("nan"),
            "std": float("nan"), "dynamic_range": float("nan"), "block_std": float("nan"),
            "structure_ratio": float("nan"), "nonfinite_frac": 1.0, "coverage_frac": 0.0,
        }

    h, w = arr.shape[:2]
    gy = np.array_split(np.arange(h), min(_BLOCK_GRID, h))
    gx = np.array_split(np.arange(w), min(_BLOCK_GRID, w))
    filled = np.where(finite_mask, arr, float(finite.mean()))
    blocks = [(ry, rx) for ry in gy for rx in gx if ry.size and rx.size]
    block_means = [float(filled[np.ix_(ry, rx)].mean()) for ry, rx in blocks]
    block_std = float(np.std(block_means))
    mean_block_px = float(np.mean([ry.size * rx.size for ry, rx in blocks]))
    std = float(finite.std())
    noise_floor = std / np.sqrt(max(mean_block_px, 1.0))
    structure_ratio = float(block_std / noise_floor) if noise_floor > 1e-12 else 0.0

    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": std,
        "dynamic_range": float(np.percentile(finite, 99) - np.percentile(finite, 1)),
        "block_std": block_std,
        "structure_ratio": structure_ratio,
        "nonfinite_frac": float(1.0 - finite_mask.mean()),
        # Fraction of the frame in the top decile of activation: how focused the map is.
        "coverage_frac": float((finite >= 0.9).mean()),
    }


def _validate_heatmap(
    heatmap: np.ndarray, method: str, strict: bool, detail: str = ""
) -> tuple[dict[str, float], list[str]]:
    """Reject or flag a heatmap that is not a real explanation.

    Non-finite values always raise: a NaN map is never a valid measurement. A
    constant map (which includes all-zero) raises under `strict`, because handing a
    reviewer a flat picture labelled "explanation" is worse than an error. Low but
    non-zero contrast warns rather than raises - it is weak evidence, not no evidence.
    """
    arr = np.asarray(heatmap, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise DegenerateExplanationError(
            f"{method}: expected a non-empty 2-D heatmap, got shape {arr.shape}."
        )

    stats = heatmap_stats(arr)
    suffix = f" {detail}" if detail else ""

    if stats["nonfinite_frac"] > 0.0:
        # Report the count as well: a single Inf in a 640x640 map is "0.0%" and the
        # bare percentage reads like a rounding artefact rather than a hard failure.
        bad = int(round(stats["nonfinite_frac"] * arr.size))
        raise DegenerateExplanationError(
            f"{method}: heatmap contains NaN or Inf ({bad} of {arr.size} pixels, "
            f"{stats['nonfinite_frac'] * 100:.2f}%). Refusing to present it as an "
            f"explanation.{suffix}"
        )

    notes: list[str] = []
    if stats["max"] - stats["min"] <= 0.0:
        message = (
            f"{method}: heatmap is perfectly uniform (constant value "
            f"{stats['mean']:.4f}); it explains nothing.{suffix}"
        )
        if strict:
            raise DegenerateExplanationError(
                message + " Pass strict=False to receive the flat map anyway."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        notes.append(message)
        return stats, notes

    if stats["std"] < _MIN_STD:
        message = (
            f"{method}: heatmap contrast is negligible (std={stats['std']:.4f} < "
            f"{_MIN_STD}); treat it as no explanation.{suffix}"
        )
        if strict:
            raise DegenerateExplanationError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        notes.append(message)
    elif stats["dynamic_range"] < _MIN_DYNAMIC_RANGE:
        message = (
            f"{method}: heatmap is near-uniform (p99-p01="
            f"{stats['dynamic_range']:.4f}).{suffix}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        notes.append(message)

    if stats["structure_ratio"] < _MIN_STRUCTURE_RATIO:
        message = (
            f"{method}: heatmap has no spatial structure (structure_ratio="
            f"{stats['structure_ratio']:.2f} < {_MIN_STRUCTURE_RATIO}); the variation "
            f"is pixel noise, not localisation.{suffix}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        notes.append(message)

    return stats, notes


# -------------------------------------------------------------------- EigenCAM


def _core_module(detector: DefectDetector) -> nn.Module:
    """Reach the raw `DetectionModel` inside the ultralytics wrapper."""
    core = getattr(getattr(detector, "_model", None), "model", None)
    if not isinstance(core, nn.Module) or not hasattr(core, "model"):
        raise TypeError(
            "Could not find the backbone module inside the detector "
            f"({type(detector).__name__}); expected detector._model.model to be an "
            "ultralytics DetectionModel."
        )
    return core


def _model_stride(core: nn.Module) -> int:
    """Output stride of the detection head, used to size the letterbox padding."""
    stride = getattr(core, "stride", None)
    if stride is None:
        return 32
    try:
        return int(max(1, int(torch.as_tensor(stride).max().item())))
    except (TypeError, ValueError, RuntimeError):
        return 32


def _resolve_layers(core: nn.Module, layer: Any) -> tuple[list[nn.Module], str]:
    """Accept None / int index / dotted name / Module / a sequence of those."""
    if layer is None:
        layer = DEFAULT_CAM_LAYER

    items = layer if isinstance(layer, (list, tuple)) else [layer]
    modules: list[nn.Module] = []
    labels: list[str] = []

    for item in items:
        if isinstance(item, nn.Module):
            modules.append(item)
            labels.append(type(item).__name__)
        elif isinstance(item, (int, np.integer)):
            blocks = core.model
            if not -len(blocks) <= int(item) < len(blocks):
                raise IndexError(
                    f"Layer index {item} out of range for a model with "
                    f"{len(blocks)} blocks (valid: 0..{len(blocks) - 1})."
                )
            modules.append(blocks[int(item)])
            labels.append(f"model.{int(item) % len(blocks)}")
        elif isinstance(item, str):
            node: Any = core
            for part in item.split("."):
                node = node[int(part)] if part.isdigit() else getattr(node, part)
            if not isinstance(node, nn.Module):
                raise TypeError(f"'{item}' resolved to {type(node)!r}, not an nn.Module.")
            modules.append(node)
            labels.append(item)
        else:
            raise TypeError(
                f"Unsupported layer specifier {item!r}; use an int index, a dotted "
                "name, an nn.Module, or a list of those."
            )
    return modules, "+".join(labels)


def _sign_corrected_eigencam_class():
    """Build the EigenCAM subclass lazily so importing this module stays cheap."""
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.svd_on_activations import (
        get_2d_projection_with_sign_correction,
    )

    class _SignCorrectedEigenCAM(EigenCAM):
        """EigenCAM with the SVD sign ambiguity resolved before the negative clip."""

        def get_cam_image(
            self, input_tensor, target_layer, target_category, activations, grads, eigen_smooth
        ):
            return get_2d_projection_with_sign_correction(activations)

    return _SignCorrectedEigenCAM


def generate_cam(
    detector: DefectDetector,
    image: str | Path | np.ndarray | Any,
    layer: Any = None,
    *,
    strict: bool = True,
    return_explanation: bool = False,
) -> np.ndarray | Explanation:
    """EigenCAM attention map for `image`, in original-image resolution.

    Returns an (H, W) float32 array in [0, 1], where H, W are the dimensions of the
    input image. The image is letterboxed to the detector's `imgsz` exactly as the
    detector letterboxes it, the CAM is computed on that canvas, and the padding is
    cropped off before the map is resized back - so a pixel in the returned map lines
    up with the same pixel in the input frame even for non-square strip captures.

    `layer` defaults to `DEFAULT_CAM_LAYER`; see that constant for why. Pass an int
    index, a dotted name such as "model.15", an `nn.Module`, or a list of those to
    average several layers.

    Raises `DegenerateExplanationError` if the result is NaN, flat or contrast-free.
    Set `return_explanation=True` to get an `Explanation` with stats and provenance
    instead of the bare array.
    """
    t0 = time.perf_counter()
    rgb = _to_rgb(image)
    core = _core_module(detector)
    modules, label = _resolve_layers(core, layer)

    canvas, geom = _letterbox(rgb, int(detector.imgsz), _model_stride(core))
    tensor = (
        torch.from_numpy(canvas.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(detector.device)
    )

    was_training = core.training
    core.eval()
    cam_class = _sign_corrected_eigencam_class()
    # `BaseCAM.__exit__` returns True for an IndexError, so the library *swallows*
    # that one exception class, prints a line to stdout and lets the `with` block
    # fall through with nothing assigned. Without this sentinel the next statement
    # would fail with an UnboundLocalError naming `raw`, which tells a caller
    # nothing and is not a DegenerateExplanationError, so `explain(method="auto")`
    # would not catch it either.
    raw: np.ndarray | None = None
    try:
        with torch.no_grad():
            # targets=[] rather than None: EigenCAM ignores targets, but the base
            # class would otherwise try to argmax the detection head's tuple output.
            with cam_class(model=core, target_layers=modules) as cam:
                raw = cam(input_tensor=tensor, targets=[])
    finally:
        if was_training:
            core.train()

    if raw is None:
        raise DegenerateExplanationError(
            f"EigenCAM[{label}] produced no map: pytorch_grad_cam suppressed an "
            "IndexError inside the CAM (it prints the message to stdout and returns "
            "control silently). Check that the target layer is a real activation "
            "site for this input size."
        )

    heatmap = _unletterbox(_normalise(raw[0]), geom)
    # Interpolation after cropping can shift the range slightly; re-normalise so the
    # contract ("float 0-1") holds exactly.
    heatmap = _normalise(heatmap)
    stats, notes = _validate_heatmap(heatmap, f"EigenCAM[{label}]", strict)

    if not return_explanation:
        return heatmap
    return Explanation(
        heatmap=heatmap,
        method="eigencam",
        layer=label,
        target=None,
        stats=stats,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        warnings=notes,
    )


# --------------------------------------------------------------------- overlay


def _draw_legend(canvas: np.ndarray, colormap: int) -> None:
    """Stamp a low-to-high colour ramp in the top-right so a saved overlay, detached
    from this code, still says which end of the scale is hot."""
    h, w = canvas.shape[:2]
    bar_w, bar_h = max(40, min(w // 4, 160)), max(6, h // 40)
    pad = max(4, h // 60)
    x0, y0 = w - bar_w - pad, pad
    if x0 <= 0 or y0 + bar_h >= h:
        return
    ramp = np.tile(np.linspace(0, 255, bar_w, dtype=np.uint8), (bar_h, 1))
    coloured = cv2.cvtColor(cv2.applyColorMap(ramp, colormap), cv2.COLOR_BGR2RGB)
    canvas[y0 : y0 + bar_h, x0 : x0 + bar_w] = coloured
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)


def overlay_cam(
    image: str | Path | np.ndarray | Any,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    *,
    weight_by_intensity: bool = True,
    colormap: int | None = None,
    draw_peak: bool = False,
    legend: bool = False,
) -> np.ndarray:
    """Blend a heatmap over the frame and return HWC RGB uint8.

    `weight_by_intensity` scales the blend by the local heatmap value, so cold
    regions stay as clean steel instead of being washed blue. That matters on mill
    imagery: a flat 0.45 blend over a light grey strip hides the very texture the
    operator is being asked to judge.
    """
    rgb = _to_rgb(image)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"Expected a 2-D heatmap, got shape {hm.shape}.")
    if not np.isfinite(hm).all():
        raise DegenerateExplanationError("Refusing to overlay a heatmap containing NaN/Inf.")

    h, w = rgb.shape[:2]
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.clip(hm, 0.0, 1.0)

    if colormap is None:
        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    coloured = cv2.applyColorMap((hm * 255.0).astype(np.uint8), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32)

    a = float(np.clip(alpha, 0.0, 1.0))
    weight = (a * hm)[..., None] if weight_by_intensity else np.full((h, w, 1), a, np.float32)
    blended = rgb.astype(np.float32) * (1.0 - weight) + coloured * weight
    out = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_peak:
        y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
        radius = max(3, int(round(max(h, w) / 40.0)))
        cv2.circle(out, (int(x), int(y)), radius, (255, 255, 255), max(1, radius // 4), cv2.LINE_AA)
    if legend:
        _draw_legend(out, colormap)
    return out


# ---------------------------------------------------------- occlusion sensitivity


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _target_score(detections: Sequence[Detection], target: Detection, iou_match: float) -> float:
    """Best confidence among detections that still describe `target`.

    Same class and overlapping the reference box: this is what "the model still sees
    that defect" means. `iou_match` is intentionally loose (0.1) because occluding
    part of a defect legitimately shrinks and shifts its box.
    """
    scores = [
        d.confidence
        for d in detections
        if d.class_id == target.class_id and _iou(d.bbox_xyxy, target.bbox_xyxy) >= iou_match
    ]
    return float(max(scores)) if scores else 0.0


def _window_origins(extent: int, size: int, stride: int) -> list[int]:
    """Window origins covering `extent`, last window flush to the far edge."""
    if extent <= size:
        return [0]
    origins = list(range(0, extent - size + 1, stride))
    if origins[-1] != extent - size:
        origins.append(extent - size)
    return origins


def occlusion_sensitivity(
    detector: DefectDetector,
    image: str | Path | np.ndarray | Any,
    patch: int = 48,
    stride: int = 24,
    *,
    target: Detection | int | None = None,
    baseline_value: int = _PAD_VALUE,
    iou_match: float = 0.1,
    batch_size: int = 16,
    strict: bool = True,
    return_explanation: bool = False,
) -> np.ndarray | Explanation:
    """Causal saliency by occlusion: how much does covering each region cost?

    A `patch` x `patch` square of flat grey is slid across the frame on `stride`
    steps. For every position the detector is re-run and the target detection's
    surviving confidence is measured; the drop from the un-occluded confidence is
    accumulated into every pixel the patch covered and averaged over the overlapping
    windows. The returned (H, W) float32 map is normalised to [0, 1], where 1.0 is
    the position that hurt the detection most.

    Cost is one forward pass per window position - 64 for a 200x200 NEU-DET frame at
    the defaults, 36 at patch=64/stride=32 - batched `batch_size` at a time. Slow, but
    it needs no gradients, no layer choice and no assumptions about the architecture,
    and unlike a CAM it is a direct causal measurement.

    The count grows as `(H/stride) * (W/stride)`, so it is the caller's job to raise
    `patch` and `stride` on a large frame: a 2600x1400 mill capture at the defaults is
    6264 passes, tens of minutes. Memory does not grow with it - occluded copies are
    built one `batch_size` chunk at a time.

    `target` selects which detection to explain: `None` for the highest-confidence
    one, an int index into `detector.predict(image).detections` (confidence-sorted),
    or a `Detection` you already hold.
    """
    t0 = time.perf_counter()
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    patch = max(1, min(int(patch), min(h, w)))
    stride = max(1, int(stride))

    if isinstance(target, Detection):
        reference = target
        baseline_conf = float(target.confidence)
    else:
        base_result = detector.predict(rgb)
        if not base_result.detections:
            raise ValueError(
                "Occlusion sensitivity needs a detection to explain, but the model "
                f"found none at conf={detector.conf}. Lower the confidence threshold "
                "or pass an explicit `target=Detection(...)`."
            )
        index = 0 if target is None else int(target)
        if not -len(base_result.detections) <= index < len(base_result.detections):
            raise IndexError(
                f"target index {index} out of range; the frame has "
                f"{len(base_result.detections)} detections."
            )
        reference = base_result.detections[index]
        baseline_conf = float(reference.confidence)

    xs = _window_origins(w, patch, stride)
    ys = _window_origins(h, patch, stride)
    boxes = [(x0, y0) for y0 in ys for x0 in xs]

    drop_sum = np.zeros((h, w), dtype=np.float32)
    hit_count = np.zeros((h, w), dtype=np.float32)
    chunk = max(1, int(batch_size))
    # Occluded copies are built one chunk at a time and dropped again. Materialising
    # the whole list first costs `n_windows * frame_bytes`, which is fine for a
    # 200 px crop (10 MB) and fatal on a mill frame: a 2600x1400 capture at the
    # default patch/stride is 6264 windows, i.e. 68 GB.
    for start in range(0, len(boxes), chunk):
        group = boxes[start : start + chunk]
        variants: list[np.ndarray] = []
        for x0, y0 in group:
            occluded = rgb.copy()
            occluded[y0 : y0 + patch, x0 : x0 + patch] = baseline_value
            variants.append(occluded)
        for (x0, y0), result in zip(group, detector.predict_batch(variants, batch_size=chunk)):
            surviving = _target_score(result.detections, reference, iou_match)
            drop = max(0.0, baseline_conf - surviving)
            drop_sum[y0 : y0 + patch, x0 : x0 + patch] += drop
            hit_count[y0 : y0 + patch, x0 : x0 + patch] += 1.0

    mean_drop = np.divide(drop_sum, hit_count, out=np.zeros_like(drop_sum), where=hit_count > 0)
    peak_drop = float(mean_drop.max())
    heatmap = _normalise(mean_drop)

    detail = (
        f"(target={reference.class_name} conf={baseline_conf:.3f}, "
        f"{len(boxes)} occlusions, peak confidence drop {peak_drop:.3f})"
    )
    stats, notes = _validate_heatmap(heatmap, "occlusion", strict, detail)
    stats["baseline_confidence"] = baseline_conf
    stats["peak_confidence_drop"] = peak_drop
    stats["n_occlusions"] = float(len(boxes))

    if not return_explanation:
        return heatmap
    return Explanation(
        heatmap=heatmap,
        method="occlusion",
        layer=None,
        target={
            "class_name": reference.class_name,
            "class_id": int(reference.class_id),
            "confidence": round(baseline_conf, 4),
            "bbox_xyxy": [round(float(v), 2) for v in reference.bbox_xyxy],
        },
        stats=stats,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        warnings=notes,
    )


# ------------------------------------------------------------------ front door


def explain(
    detector: DefectDetector,
    image: str | Path | np.ndarray | Any,
    method: str = "eigencam",
    **kwargs: Any,
) -> Explanation:
    """Produce an `Explanation` by the named method.

    "eigencam"  - fast (a single forward pass), class-agnostic attention.
    "occlusion" - slow, causal, per-detection.
    "auto"      - EigenCAM, falling back to occlusion if the CAM comes back
                  degenerate. Use it in the UI where an error dialog is worse than a
                  slower answer.
    """
    method = method.lower()
    if method == "eigencam":
        return generate_cam(detector, image, return_explanation=True, **kwargs)
    if method == "occlusion":
        return occlusion_sensitivity(detector, image, return_explanation=True, **kwargs)
    if method == "auto":
        try:
            return generate_cam(detector, image, return_explanation=True, **kwargs)
        except DegenerateExplanationError as exc:
            warnings.warn(
                f"EigenCAM was degenerate ({exc}); falling back to occlusion "
                "sensitivity.",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                return occlusion_sensitivity(detector, image, return_explanation=True)
            except (ValueError, IndexError, DegenerateExplanationError) as fallback_exc:
                # A degenerate CAM usually means a featureless frame, which also has
                # no detection for occlusion to explain. Surfacing the fallback's
                # "no detections" error would hide the real problem.
                raise DegenerateExplanationError(
                    f"{exc} Occlusion fallback also failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
    raise ValueError(f"Unknown method {method!r}; expected 'eigencam', 'occlusion' or 'auto'.")


# ----------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    """The CLI documented in README.md under "Other entry points".

    Split out of the ``__main__`` block on purpose. The argument-defaulting bug
    this replaced (``load_detector(imgsz=args.imgsz)`` running one line *before*
    the ``None`` fallback, so the documented command died with
    ``TypeError: int() argument must be a string ... not 'NoneType'``) survived
    because everything below ``if __name__ == "__main__"`` was marked
    ``# pragma: no cover`` and no test could reach it. A parser and a ``main``
    that are importable are testable.
    """
    parser = argparse.ArgumentParser(description="Explain one frame.")
    parser.add_argument("image")
    parser.add_argument("--method", default="eigencam", choices=["eigencam", "occlusion", "auto"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--out", default=None, help="write the overlay PNG here")
    parser.add_argument("--weights", default=None, help="checkpoint (default: inference.resolve_weights)")
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help="network input size (default: inference.DEFAULT_IMGSZ). A CAM is an "
             "explanation of a specific forward pass, so it must be produced at the "
             "input size the model is actually served at.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Explain one frame and print the JSON record to stdout."""
    args = build_parser().parse_args(argv)

    # imgsz is resolved by the parser default, never left as None: `load_detector`
    # casts it with `int()` and a None here is the crash this CLI used to ship.
    imgsz = DEFAULT_IMGSZ if args.imgsz is None else int(args.imgsz)
    detector = load_detector(weights=args.weights, imgsz=imgsz)

    kw: dict[str, Any] = {}
    if args.method != "occlusion" and args.layer is not None:
        kw["layer"] = args.layer
    exp = explain(detector, args.image, method=args.method, **kw)
    print(json.dumps(exp.to_dict(), indent=2))
    if args.out:
        overlay = overlay_cam(args.image, exp.heatmap)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"cv2 refused to write {args.out}")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin wrapper over main()
    raise SystemExit(main())
