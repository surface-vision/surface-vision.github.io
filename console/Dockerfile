# Hugging Face Spaces no longer offers a native `streamlit` SDK -- the valid
# options are gradio, docker and static -- so this Space is a docker Space that
# runs streamlit itself. Everything the streamlit SDK used to do implicitly is
# done explicitly here.
FROM python:3.11-slim

# Same two libraries packages.txt documents: ultralytics declares opencv-python by
# name while requirements.txt pins opencv-python-headless, both wheels own the
# same cv2/ directory, and whichever pip unpacks last wins. Installing these makes
# `import cv2` succeed either way instead of failing on libGL.so.1 at startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000. Files copied as root would be unwritable
# by the app, and streamlit needs a writable home for its config and cache.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # No GPU on the free CPU tier; make that explicit rather than letting torch probe.
    CUDA_VISIBLE_DEVICES="" \
    # Ultralytics writes settings and font downloads under $HOME by default; point
    # them somewhere writable so a read-only path cannot stall first inference.
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib

WORKDIR /app

# Requirements first so the dependency layer caches across app edits. The
# --extra-index-url that resolves the +cpu torch wheels lives inside the file.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

# Spaces routes public traffic to 7860.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
