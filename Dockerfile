# PyTorch base image for CLIP
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app.py ./

ENV AWS_REGION=us-east-1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


