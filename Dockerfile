# Python 3.10 + Torch 2.2.2 + CUDA 12.1 (nvcc available but we won't need it)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/root/.cache/huggingface

# System basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# App
COPY app.py ./

# Default config (override in ECS/EC2 as needed)
ENV AWS_REGION=us-east-1 \
    FLORENCE_MODEL_ID=microsoft/Florence-2-large-ft \
    SIGLIP_MODEL_ID=google/siglip-so400m-patch14-384 \
    TOP_K=8

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
