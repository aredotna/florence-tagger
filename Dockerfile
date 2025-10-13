# FINAL: Florence + flash-attn (builds from source)
# Python 3.10 + Torch 2.2.2 + CUDA 12.1 (has nvcc)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Build tools (nvcc already in this image)
RUN apt-get update && apt-get install -y git build-essential ninja-build && rm -rf /var/lib/apt/lists/*

# Build settings for A10G (compute capability 8.6). Adjust if using a different GPU.
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86"
ENV MAX_JOBS=4

# Pip basics
RUN pip install --upgrade pip setuptools wheel

# ---- Install flash-attn (this is the step that needs nvcc) ----
# We pin to a version that’s compatible with torch 2.2.x/cu121/py310
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.5.6

# ---- App deps (no torch here) ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

# App
COPY app.py ./

ENV AWS_REGION=us-east-1 \
    FLORENCE_MODEL_ID=microsoft/Florence-2-base \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
