#!/usr/bin/env python3
"""Check that hf_space/vendor/*.py are still copies of the repository's src/*.py.

A vendored file is a copy that will drift, so the drift has to be detectable. This
compares each file in `vendor/` against its upstream, ignoring exactly two things:
the `VENDORED COPY` header block that is inserted into every file, and the marked
`# SPACE:` deviation in `calibrate.py`. Anything else that differs is reported as a
diff and exits non-zero.

It also asserts the one constant `vendor/calibrate.py` mirrors from a module it no
longer imports, so that value cannot silently go stale.

    python hf_space/tools/verify_vendor.py [--repo /path/to/repo]

Needs the repository, so it is a pre-push check on a developer's machine, not
something the Space runs. Exit status: 0 clean, 1 drift, 2 cannot compare.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

SPACE_ROOT = Path(__file__).resolve().parent.parent
MODULES = ("inference", "domain_shift", "ood_guard", "calibrate", "explain", "report")

HEADER_START = "# ---------------------------------------------------------------------------\n# VENDORED COPY"
HEADER_END = "# ---------------------------------------------------------------------------\n"

# The one marked deviation: vendor/calibrate.py replaces upstream's module-scope
# `from evaluate import ...` block. Both sides are matched by pattern rather than by
# literal text so a reworded comment inside the region does not fail the check.
CALIBRATE_UPSTREAM = re.compile(
    r"try:  # works both as `python src/calibrate\.py`.*?"
    r"from inference import CLASS_NAMES, DefectDetector, resolve_device, resolve_weights  # noqa: E402\n",
    re.DOTALL,
)
CALIBRATE_VENDOR = re.compile(
    r"sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parent\)\)\n\n"
    r"from inference import \(  # noqa: E402\n.*?"
    r"    CachedPrediction = GroundTruth = object\n",
    re.DOTALL,
)
PLACEHOLDER = "<<< IMPORT BLOCK >>>\n"


def strip_header(text: str) -> str:
    """Remove the inserted VENDORED COPY comment block."""
    start = text.find(HEADER_START)
    if start == -1:
        return text
    end = text.find(HEADER_END, start + len(HEADER_START))
    if end == -1:
        return text
    return text[:start] + text[end + len(HEADER_END):]


def normalise(module: str, upstream: str, vendored: str) -> tuple[str, str]:
    vendored = strip_header(vendored)
    if module == "calibrate":
        upstream, n_up = CALIBRATE_UPSTREAM.subn(PLACEHOLDER, upstream, count=1)
        vendored, n_ve = CALIBRATE_VENDOR.subn(PLACEHOLDER, vendored, count=1)
        if n_up != 1 or n_ve != 1:
            raise SystemExit(
                f"[verify_vendor] cannot locate the marked import block in "
                f"calibrate.py (upstream {n_up}, vendored {n_ve}). Re-check the "
                f"deviation by hand rather than trusting this script."
            )
    return upstream, vendored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=SPACE_ROOT.parent,
        help="repository root that holds src/ (default: the Space's parent)",
    )
    args = parser.parse_args(argv)
    src = args.repo / "src"
    if not src.is_dir():
        print(f"[verify_vendor] no src/ at {src} -- nothing to compare against.")
        return 2

    failures = 0
    for module in MODULES:
        up_path, ve_path = src / f"{module}.py", SPACE_ROOT / "vendor" / f"{module}.py"
        if not ve_path.is_file():
            print(f"FAIL  {module}: not vendored")
            failures += 1
            continue
        if not up_path.is_file():
            print(f"FAIL  {module}: no upstream at {up_path}")
            failures += 1
            continue
        upstream, vendored = normalise(
            module, up_path.read_text(), ve_path.read_text()
        )
        if upstream == vendored:
            print(f"ok    {module}.py  ({len(vendored.splitlines())} lines)")
            continue
        failures += 1
        print(f"FAIL  {module}.py drifted from {up_path}:")
        diff = difflib.unified_diff(
            upstream.splitlines(True), vendored.splitlines(True),
            fromfile=f"src/{module}.py", tofile=f"vendor/{module}.py", n=1,
        )
        sys.stdout.writelines(list(diff)[:60])

    # The constant vendor/calibrate.py mirrors out of a module it no longer imports.
    evaluate = src / "evaluate.py"
    if evaluate.is_file():
        match = re.search(r"^IOU_MATCH = ([0-9.]+)", evaluate.read_text(), re.M)
        vendor_match = re.search(
            r"^    IOU_MATCH = ([0-9.]+)",
            (SPACE_ROOT / "vendor" / "calibrate.py").read_text(),
            re.M,
        )
        if not match or not vendor_match:
            print("FAIL  IOU_MATCH: could not read it from both sides")
            failures += 1
        elif match.group(1) != vendor_match.group(1):
            print(
                f"FAIL  IOU_MATCH: src/evaluate.py has {match.group(1)}, "
                f"vendor/calibrate.py mirrors {vendor_match.group(1)}"
            )
            failures += 1
        else:
            print(f"ok    IOU_MATCH mirrored correctly ({match.group(1)})")

    print(
        f"\n[verify_vendor] {len(MODULES)} module(s) checked, {failures} problem(s)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
