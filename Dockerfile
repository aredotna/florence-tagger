FROM ghcr.io/dao-aillab/flash-attention:2.5.6-cu121

# This base already has CUDA 12.1 + flash-attn compiled in.
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR /app

# torch is already included
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY app.py ./

ENV FLORENCE_MODEL_ID=microsoft/Florence-2-base
ENV AWS_REGION=us-east-1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
