FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
# Install base deps (Torch first)
RUN pip3 install --no-cache-dir -r requirements.txt

# Install flash-attn (prebuilt wheel) AFTER torch is present
# The --no-build-isolation flag helps pip pick the correct wheel without compiling.
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn==2.5.6

# Download spaCy English model
RUN python3 -m spacy download en_core_web_sm

COPY app.py ./
EXPOSE 8000
ENV AWS_REGION=us-east-1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
