# Python 3.10 + Torch 2.2.2 + CUDA 12.1
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git build-essential ninja-build && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
    MAX_JOBS=4

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add OpenAI CLIP + its tokenizer deps
RUN pip install --no-cache-dir \
    "git+https://github.com/openai/CLIP.git" \
    ftfy==6.2.0 \
    regex==2024.7.24

# Install the official RAM code (gives us the ram.* modules and inference helpers)
RUN pip install --no-cache-dir git+https://github.com/xinyu1205/recognize-anything.git

# OPTIONAL earlier — now REQUIRED by Florence remote code:
ENV TORCH_CUDA_ARCH_LIST="8.6" CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" MAX_JOBS=4
RUN pip install --no-cache-dir --no-build-isolation "flash-attn==2.5.6"

# Pre-download RAM checkpoints from HF into the image layer
# (RAM base: works well; if you want RAM++, swap to the plus checkpoints — see notes below)
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
# RAM (base) weights + tag embeddings
hf_hub_download("xinyu1205/recognize_anything_model", "ram_swin_large_14m.pth", local_dir="/models/ram")
hf_hub_download("xinyu1205/recognize_anything_model", "ram_tag_embedding_class_4585.pth", local_dir="/models/ram")
PY

# spaCy (if you still need it; optional)
# RUN python -m spacy download en_core_web_sm

# App
COPY app.py ./

ENV AWS_REGION=us-east-1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    FLORENCE_MODEL_ID=microsoft/Florence-2-large-ft \
    RAM_VARIANT=ram             \
    RAM_WEIGHTS=/models/ram/ram_swin_large_14m.pth \
    RAM_TAG_EMB=/models/ram/ram_tag_embedding_class_4585.pth

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


