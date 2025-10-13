FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

WORKDIR /app

# 1) Install PyTorch CUDA
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# 2) Other dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 3) Flash-attn fallback to satisfy import
RUN pip install --no-cache-dir flash-attn --no-build-isolation --no-deps || \
    pip install --no-cache-dir git+https://github.com/Dao-AILab/flash-attention.git@main#egg=flash-attn --no-deps || true

# 4) spaCy model
RUN python3 -m spacy download en_core_web_sm

COPY app.py ./
EXPOSE 8000
ENV AWS_REGION=us-east-1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
