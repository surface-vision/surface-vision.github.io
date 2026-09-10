# Deploying this directory as a Hugging Face Space

This directory is a complete, self-contained Space. It has not been pushed: at the
time it was built there was no Hugging Face account or token for it. Everything
below assumes a token arrives later.

Nothing here needs editing before a push. The only value that has to be chosen is
the Space id -- `<owner>/<name>` -- which appears in every command as
`$SPACE_ID`.

    export SPACE_ID="your-username/surface-inspection-console"
    export HF_TOKEN="hf_..."          # a WRITE token: hf.co/settings/tokens
    export SPACE_DIR="/Users/.../hf_space"   # this directory, absolute

---

## 0. Before either path: Git LFS

`.gitattributes` here tracks `*.pt`, `*.onnx`, `*.pth`, `*.bin` and
`*.safetensors` through LFS, which is the Hub convention. **If LFS is not
installed in the pushing clone, both 6 MB checkpoints arrive on the Hub as
~130-byte pointer text files and the Space starts with "No usable detector
checkpoint was found".** That failure has a named screen in `app.py` precisely
because it is the likeliest way this deployment breaks.

    git lfs install          # once per machine
    git lfs version          # confirm it answers

The `huggingface_hub` path (A) handles large files itself and does not need
this; the plain git path (B) does.

---

## Path A -- `huggingface_hub` from Python

Recommended. It creates the Space with the right SDK and uploads the folder in one
call, and it does not care whether git-lfs is installed.

    pip install "huggingface_hub>=0.34"

```python
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
space_id = os.environ["SPACE_ID"]

# 1. create the Space. space_sdk must match the README front matter.
api.create_repo(
    repo_id=space_id,
    repo_type="space",
    space_sdk="streamlit",
    private=False,          # make it public when you want it seen
    exist_ok=True,          # safe to re-run
)

# 2. upload everything. README.md carries the Space configuration in its YAML
#    front matter, so it must be included -- it is not documentation here, it is
#    the config file.
api.upload_folder(
    repo_id=space_id,
    repo_type="space",
    folder_path=os.environ["SPACE_DIR"],
    commit_message="Surface inspection console: Streamlit console, CPU-only, vendored inference stack",
    ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store", ".git/*"],
)

print(f"https://huggingface.co/spaces/{space_id}")
```

Same thing from the shell, with the `hf` CLI that ships with `huggingface_hub`:

    hf auth login --token "$HF_TOKEN"
    hf repo create "$SPACE_ID" --repo-type space --space-sdk streamlit
    hf upload "$SPACE_ID" "$SPACE_DIR" . --repo-type space \
        --commit-message "Surface inspection console" \
        --exclude "__pycache__/*" --exclude "*.pyc" --exclude ".DS_Store"

(On `huggingface_hub` older than 1.0 the CLI is `huggingface-cli` and the
subcommands are `login`, `repo create`, `upload` with the same arguments.)

---

## Path B -- plain git remote

Use this when the Space should have real git history, or when the push has to
happen from CI with nothing but git.

    # 1. create the Space first -- an empty git push cannot choose an SDK.
    #    Either use the web form at https://huggingface.co/new-space
    #    (SDK: Streamlit) or the one API call from Path A.

    cd "$SPACE_DIR"
    git lfs install
    git init
    git add -A
    git commit -m "Surface inspection console: Streamlit console, CPU-only, vendored inference stack"
    git branch -M main

    # 2. the token goes in the remote URL. Use a WRITE token.
    git remote add space "https://USER:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}"
    git push space main

If the Space was created through the web form it already has a commit (its own
README), so the first push is rejected as non-fast-forward. Either pull and
rebase onto it --

    git pull --rebase space main       # then resolve README.md in favour of THIS one
    git push space main

-- or, because that generated README is only a placeholder, overwrite it:

    git push --force space main

**Do not let the Hub's generated README survive.** Its front matter is what
configures the Space, and this directory's version is the one with the correct
`sdk_version`, `python_version` and `app_file`.

Confirm the checkpoints went up as LFS objects and not as pointers:

    git lfs ls-files
    # expect two lines: models/yolov8n_joint/weights/best.pt
    #                   models/yolov8n_neudet/weights/best.pt

---

## What the Space does at build time

1. `packages.txt` -> apt installs `libgl1` and `libglib2.0-0`.
2. `requirements.txt` -> pip, with `--extra-index-url https://download.pytorch.org/whl/cpu`
   on its first line so torch resolves to the CPU wheel.
3. `streamlit run app.py` on port 7860, from the repository root.

Resolved footprint for the Linux/cp311 branch of `requirements.txt`, measured
against the live indexes: **69 wheels plus one sdist, 579.8 MB of downloads**, of
which torch is 196.2 MB. There is **no CUDA payload**: `torch 2.14.0+cpu`
declares no `nvidia-*`, `cuda-*` or `triton` dependency at all, where the plain
PyPI `torch==2.14.0` declares seven of them under `platform_system == "Linux"`
(cuda-toolkit 13.0.3 with eleven extras, cuda-bindings, nvidia-cudnn-cu13,
nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13 and triton) --
1.30 GB of wheels before the cuda-toolkit extras are expanded. That single line
is the difference between a Space that builds and one that times out.

`grad-cam==1.5.7` is published as an sdist only (44 KB, pure Python), so pip
builds one wheel during the install. That is expected, not a fault.

---

## After the push -- check these four things

1. **Build log** (Space page -> Logs -> Build). `Successfully installed ...` should
   list `torch-2.14.0+cpu`. If it lists `torch-2.14.0` with `nvidia-*` packages,
   the `--extra-index-url` line was lost -- check it is still the first
   non-comment line of `requirements.txt`.
2. **Runtime log** (-> Logs -> Container). Expect Streamlit's startup lines and
   nothing else. `ImportError: libGL.so.1` means `packages.txt` did not apply;
   see its comments for the fallback.
3. **The page itself.** It must open on a held-out frame with a verdict already
   rendered, the header rail reading `Model yolov8n_neudet/best.pt`,
   `Compute CPU`, `Mode AUTO`, `Classes 6`. If the checkpoint picker is empty,
   the weights are LFS pointers -- see section 0.
4. **Frame source -> Wide strip capture.** The badge must read `TILED` and the
   detection count 20. That single click exercises decode, the gate, the tiling
   decision and 50 tiled forward passes, which is most of the stack.

Expect the first page load after a cold start to take longer than the numbers in
`README.md`: the container has to import torch, build a matplotlib font cache and
warm the detector. On the development host, from a cold process, a full script run
through the real server measured 3.0 s once those caches existed and 12.1 s when
they did not.

---

## Keeping it in sync with the repository

`vendor/*.py` are copies, so they will drift. Before any later push:

    python tools/verify_vendor.py --repo /path/to/the/project

It diffs every vendored module against its upstream, ignoring only the inserted
`VENDORED COPY` header and the one marked deviation in `calibrate.py`, and checks
the single constant that file mirrors out of a module it no longer imports. Clean
output is seven `ok` lines and `0 problem(s)`.

To re-measure the latency table in `README.md` on the Space's own hardware rather
than the development host's:

    python tools/bench_cpu.py --threads 2 --cam
