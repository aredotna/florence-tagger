FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# system deps
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

WORKDIR /app

# Install CUDA-enabled PyTorch first (cu121 wheels)
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Now the rest of the Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download spaCy English model inside the image
RUN python3 -m spacy download en_core_web_sm

# App code
COPY app.py ./

EXPOSE 8000
ENV AWS_REGION=us-east-1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
