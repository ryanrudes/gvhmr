# GVHMR gradio demo — legacy **CPU Docker** Space (local / offline testing).
#
# Production HuggingFace Spaces use the **Gradio SDK + ZeroGPU** path (`app.py` + `requirements.txt`;
# see space/README.md). Docker cannot use ZeroGPU.
#
# Why this Dockerfile exists: `chumpy` (legacy SMPL .pkl loader) ships a setup.py that does
# `import pip` at build time. The Gradio SDK builder uses PEP 517 isolation (no pip in the build
# env), so we bootstrap chumpy in `app.py` instead. This Docker image keeps the old CPU-only path
# for developers who want `docker build` without ZeroGPU.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    GVHMR_CHECKPOINTS=/tmp/gvhmr/checkpoints \
    GVHMR_BODY_MODELS=/tmp/gvhmr/body_models \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# System libraries for OpenCV / video IO / OpenGL mesh rendering, plus a C/C++ toolchain —
# the `preproc` extra builds native wheels from sdist (e.g. cython-bbox) that need gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
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
