#!/usr/bin/env python3
"""Measure the console's resident memory, stage by stage, against a host budget.

WHY THIS EXISTS. Hugging Face now charges for Docker and Gradio Spaces on CPU, so
the free host for this console is Streamlit Community Cloud, whose container is of
the order of 1 GB -- against the 16 GB a free HF CPU Space gave. That is tight
enough for "does it fit?" to need a measurement rather than an opinion, and tight
enough that the answer has to be re-checkable after any dependency bump. Hence a
committed tool instead of a number in a README.

WHAT IT MEASURES. Resident set size of this process, sampled after each stage of a
cold boot and then after each of the three things that can grow it. The app stages
run through `streamlit.testing.v1.AppTest`, which executes `app.py` under a real
Streamlit runtime -- the same ScriptRunner, the same `st.cache_resource`, the same
session state -- so these are the app running, not an approximation of it.

    python tools/measure_memory.py             # cold boot + the growth paths
    python tools/measure_memory.py --explain    # also import grad-cam and run a CAM
    python tools/measure_memory.py --budget 1024

READING THE NUMBER. RSS on Linux, where this deploys, is what the container's
cgroup accounts and so what gets the app OOM-killed. RSS on macOS, where it is
developed, counts the same anonymous allocations but pages shared library mappings
in and out at the kernel's discretion -- `libtorch_cpu` is the bulk of this
process and it is mapped, not copied -- so the macOS figure moves by tens of MB
between identical runs and is a guide, not a guarantee. Treat the headroom, not
the total, as the result: the stages this tool bounds are anonymous allocations
that behave the same on both platforms.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

SPACE_ROOT = Path(__file__).resolve().parent.parent


class Meter:
    """RSS sampler that remembers its stages so it can print a summary."""

    def __init__(self) -> None:
        import psutil  # noqa: PLC0415  (optional dependency of this tool only)

        self.proc = psutil.Process()
        self.stages: list[tuple[str, float, float]] = []
        self.last = self._rss()

    def _rss(self) -> float:
        return self.proc.memory_info().rss / (1024 * 1024)

    def mark(self, label: str) -> float:
        gc.collect()
        rss = self._rss()
        delta = rss - self.last
        self.last = rss
        self.stages.append((label, rss, delta))
        print(f"  {rss:8.1f} MB  {delta:+8.1f}   {label}", flush=True)
        return rss

    @property
    def peak(self) -> float:
        return max((rss for _, rss, _ in self.stages), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=1024.0,
                    help="host memory budget in MB to compare against (default 1024)")
    ap.add_argument("--explain", action="store_true",
                    help="also measure the Explain button (imports pytorch_grad_cam)")
    ap.add_argument("--decode-entries", type=int, default=64,
                    help="how many wide frames to push through the decode cache")
    args = ap.parse_args()

    os.chdir(SPACE_ROOT)
    for path in (SPACE_ROOT, SPACE_ROOT / "vendor"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    m = Meter()
    print(f"budget {args.budget:.0f} MB   platform {sys.platform}   "
          f"python {sys.version.split()[0]}\n")
    print("       RSS       delta    stage")

    from streamlit.testing.v1 import AppTest  # noqa: PLC0415

    m.mark("interpreter + streamlit.testing")

    # --- cold boot ----------------------------------------------------------- #
    at = AppTest.from_file(str(SPACE_ROOT / "app.py"), default_timeout=900)
    at.run()
    if at.exception:
        print(f"\nFAIL: the script raised {[e.value for e in at.exception]}")
        return 1
    cold = m.mark("cold boot: imports, checkpoint, warmup, one frame scored")

    rendered = " ".join(x.value for x in at.markdown)
    if "DEFECT" not in rendered and "PASS" not in rendered:
        print("\nFAIL: cold boot rendered no verdict, so no inference ran")
        return 1

    # --- growth path 1: the tiled path on a real strip capture --------------- #
    if at.radio and len(at.radio[0].options) > 1:
        at.radio[0].set_value(at.radio[0].options[1]).run()
        if at.exception:
            print(f"\nFAIL: {[e.value for e in at.exception]}")
            return 1
        m.mark("2048x1000 strip through the tiled path")

    # --- growth path 2: switching checkpoint in the sidebar ------------------ #
    import inference  # noqa: PLC0415

    switched = [s for s in at.sidebar.selectbox
                if any("joint" in str(o) for o in (s.options or []))]
    if switched:
        target = [o for o in switched[0].options if o != switched[0].value][0]
        switched[0].set_value(target).run()
        if at.exception:
            print(f"\nFAIL: {[e.value for e in at.exception]}")
            return 1
        resident = len(inference._DETECTOR_CACHE)
        m.mark(f"sidebar checkpoint switch ({resident} detector(s) resident)")
        if resident != 1:
            print(f"\nFAIL: {resident} detectors resident after a switch, expected 1.\n"
                  "      get_detector's bound on inference._DETECTOR_CACHE is not holding.")
            return 1

    # --- growth path 3: the decode cache ------------------------------------- #
    import app  # noqa: PLC0415

    strip = SPACE_ROOT / "assets" / "strip_sample_edge_defect.jpg"
    if strip.exists() and args.decode_entries > 0:
        data = strip.read_bytes()
        for i in range(args.decode_entries):
            app.cached_decode(data, f"synthetic_upload_{i}.jpg")
        m.mark(f"{args.decode_entries} wide frames pushed through the decode cache")

    # --- growth path 4: the Explain button ----------------------------------- #
    if args.explain:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from explain import explain as explain_frame  # noqa: PLC0415

        frame = SPACE_ROOT / "data" / "neu-det" / "test" / "images" / "scratches_271.jpg"
        detector = next(iter(inference._DETECTOR_CACHE.values()))
        explain_frame(detector, np.array(Image.open(frame).convert("RGB")), method="auto")
        m.mark("Explain button: pytorch_grad_cam imported, one CAM")

    # --- verdict ------------------------------------------------------------- #
    peak = m.peak
    headroom = args.budget - peak
    print(f"\ncold boot {cold:.1f} MB      peak {peak:.1f} MB      "
          f"headroom {headroom:+.1f} MB of {args.budget:.0f} MB")
    if headroom <= 0:
        print("VERDICT: does NOT fit. See the MEMORY BUDGET comment in app.py.")
        return 1
    print(f"VERDICT: fits, with {100 * headroom / args.budget:.0f}% of the budget spare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
