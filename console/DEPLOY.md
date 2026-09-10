# Deploying this console

The console is a Streamlit app. This document is the click-by-click for putting it
on **Streamlit Community Cloud**, which is free, and then two self-hosting paths
for anyone who would rather run it themselves.

Everything needed is already in this repository. Nothing has to be edited, no
token has to be created, and no file has to be uploaded by hand.

---

## Why Streamlit Community Cloud and not a Hugging Face Space

This directory was built to be a Hugging Face Space, and the `Dockerfile` that
would still make it one is kept below. It is not deployed there because Hugging
Face now requires a **PRO subscription for Docker and Gradio Spaces on free CPU
hardware**; only *static* Spaces remain free, and a static Space cannot run a
Python process. That is a platform pricing policy, not a defect in this package --
the image builds and the app runs; there is simply no free CPU Space to run it on.

Streamlit Community Cloud is free, is purpose-built for exactly this kind of app,
and deploys straight from a GitHub repository. It does not use the `Dockerfile`;
it reads `requirements.txt`, `packages.txt` and `app.py` at the repository root,
all three of which are here.

---

## Path A -- Streamlit Community Cloud (the one-click path)

**What you need:** the GitHub account that owns or can access
`surface-vision/surface-vision-console`. Nothing else. No credit card, no token.

**Time:** about two minutes of clicking, then 5-12 minutes of unattended build.

### 1. Sign in

Go to **<https://share.streamlit.io>** and click **Continue with GitHub**.
GitHub will ask you to authorise the Streamlit app. Accept.

> **The one snag worth knowing about.** This repository lives in the
> `surface-vision` **organisation**, not on a personal account. GitHub OAuth apps
> get access to an organisation only when an owner grants it. On the
> authorisation screen there is a list of organisations with a **Grant** button
> next to each -- click **Grant** next to `surface-vision` before you hit
> Authorize. If you miss it, the repository simply will not appear in the
> dropdown in step 3, and the fix is
> <https://github.com/settings/connections/applications> → Streamlit →
> Organization access → **Grant** next to `surface-vision`.

### 2. Start a new app

On the workspace page, click **Create app** (top right), then choose
**Deploy a public app from GitHub**.

### 3. Fill in exactly these three fields

| Field | Value |
| --- | --- |
| **Repository** | `surface-vision/surface-vision-console` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

The **App URL** field underneath is a subdomain you can choose freely; whatever
you pick becomes `https://<that>.streamlit.app`.

### 4. Set the Python version -- do not skip this

Click **Advanced settings** and set **Python version** to **3.11**.

This is the one setting that is not the platform default and it matters:
`requirements.txt` pins `torch==2.14.0+cpu` and `torchvision==0.29.0+cpu`, and the
CPU wheels are resolved per interpreter version. On the default interpreter the
build can spend several minutes resolving before it fails, which looks like a
broken repository and is not one.

Leave **Secrets** empty. This app has none -- it reads nothing but its own files.

### 5. Click Deploy

Then watch the log pane on the right. In order you will see:

1. **Cloning the repository.** Seconds. All 13 MB of it, both checkpoints
   included, because they are ordinary git blobs and not LFS pointers -- see
   `.gitattributes`, which explains why that is deliberate.
2. **`apt-get` installing `packages.txt`.** `libgl1` and `libglib2.0-0`. Seconds.
3. **`pip` installing `requirements.txt`.** This is the slow part and it is
   normal. The CPU build of torch is a 196 MB wheel and ultralytics, pandas,
   plotly, matplotlib and grad-cam follow it. **Budget 5-12 minutes.** The log
   goes quiet during the torch download; quiet is not stuck.
4. **`streamlit run app.py`.** The app appears.

### 6. What the first screen should look like

A dark console headed **Surface Inspection Console**, a sidebar with a checkpoint
selector and three sliders, and four tabs: *Single frame*, *Batch inspection*,
*Line simulation*, *Defect atlas*. The *Single frame* tab opens with a NEU-DET
test frame already selected and already scored -- you should see an annotated
overlay and a verdict without clicking anything.

**First interaction is slower than the rest.** The first inference in a fresh
container pays lazy kernel compilation; `app.py` warms the model with two dummy
passes at boot precisely so a judge does not pay it, but a cold container is still
a cold container.

**The app sleeps.** Community Cloud suspends an app after about a week without
traffic and shows a "this app has gone to sleep" button. Anyone can press it to
wake it, but it means a cold boot. If this is going in front of judges, open it
once on the morning of.

### 7. Will it fit in the memory the free tier gives?

Yes, measured, with about 40% spare. This mattered enough to be worth a tool:

    python tools/measure_memory.py --explain

which runs `app.py` under a real Streamlit runtime, drives it through every path
that can grow the process, and exits non-zero if the peak crosses the budget.
On this build:

    cold boot: imports, checkpoint, warmup, one frame scored     461 - 495 MB
    peak, having also run the tiled path on a 2048x1000 strip,
    switched checkpoint, decoded 400 wide frames and hit Explain  585 - 618 MB
    budget                                                            ~1024 MB

Getting there took two reductions, both documented at their definitions in
`app.py`: only one detector is ever resident (`get_detector`), and decoded frames
are no longer cached through `st.cache_data`, whose pickle round-trip on a 6 MB
image array was worth 386 MB on its own (`cached_decode`). Before those, the peak
was 923 MB -- inside the noise of the limit, which is not a place to be.

Two caveats on that number, because it is a measurement and not a promise. It is
RSS on macOS; the container is Linux, where the same anonymous allocations behave
the same way but shared library mappings are accounted differently, so treat the
~400 MB of headroom as the result rather than the total. And Streamlit serves
every browser session from one process: the figures above are one session, and the
model, the calibration map and the sample catalogue are shared across all of them,
but each concurrent session holds its own decoded frame.

---

## Path B -- run it locally

    pip install -r requirements.txt
    streamlit run app.py

Python 3.11. On macOS the `+cpu` pins fall back to the ordinary PyPI wheels by
marker, so the same file works. Opens on <http://localhost:8501>.

## Path C -- self-host the container

The `Dockerfile` is unchanged and still correct. It is what a Hugging Face Docker
Space would have used, and it works anywhere that runs a container:

    docker build -t surface-console .
    docker run --rm -p 7860:7860 surface-console

Then <http://localhost:7860>. It runs as uid 1000, exposes 7860, and has a
healthcheck on `/_stcore/health`.

### If you do want to push this to a Hugging Face Space

It needs the YAML front matter that this README no longer carries, because that
front matter is meaningless on GitHub. Put this back at the very top of
`README.md` before pushing to a Space, and note that it requires a PRO
subscription to run on CPU:

    ---
    title: Surface Inspection Console
    emoji: 🔶
    colorFrom: yellow
    colorTo: gray
    sdk: docker
    app_port: 7860
    pinned: false
    short_description: Steel strip surface defect detection, with the caveats attached
    ---

On that path, and **only** on that path, tracking `*.pt` through Git LFS is the
Hub convention and is fine. It is actively wrong on the Streamlit Cloud path --
see `.gitattributes`.

---

## Troubleshooting

**The repository is not in the dropdown.** Organisation access was not granted.
<https://github.com/settings/connections/applications> → Streamlit → Organization
access → **Grant** next to `surface-vision`.

**The build fails resolving torch.** The Python version is not 3.11. Manage app →
Settings → Python version. Changing it forces a rebuild.

**`ImportError: libGL.so.1` at `import cv2`.** `packages.txt` did not apply.
It must contain nothing but package names, one per line -- no comments, no blank
lines -- because those lines are handed to `apt-get` verbatim. That is why the
explanation for `libgl1` and `libglib2.0-0` lives in this document instead of in
that file.

**"No usable detector checkpoint was found."** The `.pt` files arrived as LFS
pointer text rather than as torch archives. Check that `.gitattributes` contains
no `filter=lfs` line; a real checkpoint here is ~6.2 MB and starts with the bytes
`PK\x03\x04`.

**The app restarts itself under load.** That is the OOM killer, and it means
something has been added since the figures above were measured. Run
`python tools/measure_memory.py --explain` and read the MEMORY BUDGET comment in
`app.py`.
