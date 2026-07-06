# GVHMR gradio demo — HuggingFace **Docker** Space.
#
# Why Docker instead of the gradio SDK: a base dependency, `chumpy` (loads legacy SMPL .pkl body
# models), ships a legacy setup.py that does `import pip` at build time. The gradio SDK installs
# `requirements.txt` with PEP 517 build isolation, whose isolated build env has no `pip`, so the
# chumpy wheel build fails. Here we install chumpy first with build isolation disabled (the base
# env has pip + setuptools + numpy), then the rest — a robust, self-contained build.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    GVHMR_CHECKPOINTS=/tmp/gvhmr/checkpoints \
    GVHMR_BODY_MODELS=/tmp/gvhmr/body_models \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# System libraries for OpenCV / video IO / OpenGL mesh rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) chumpy — legacy build needs pip in the build env, so disable isolation (numpy must be present).
RUN pip install --upgrade pip setuptools wheel \
    && pip install "numpy>=1.26" \
    && pip install --no-build-isolation "chumpy==0.70"

# 2) CPU torch — the free Spaces tier is CPU-only. For a GPU Space, drop this line and add the CUDA
#    wheel index instead, e.g. `--index-url https://download.pytorch.org/whl/cu121`.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) GVHMR (from PyPI, with the preproc extra) + gradio.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
