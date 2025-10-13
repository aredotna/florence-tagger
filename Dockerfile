FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime  # Python 3.10 inside

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR /app

# flash-attn wheel exists for torch 2.2.2/cu121/py310
RUN pip install --no-cache-dir flash-attn==2.5.6 --no-build-isolation

# app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

# app
COPY app.py ./
ENV FLORENCE_MODEL_ID=microsoft/Florence-2-base
ENV AWS_REGION=us-east-1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
