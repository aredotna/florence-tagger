# GPU image; works on your g5.xlarge with NVIDIA runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# system deps
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# install python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# download spaCy English model inside the image (not via pip)
RUN python3 -m spacy download en_core_web_sm

COPY app.py ./

EXPOSE 8000
ENV AWS_REGION=us-east-1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
