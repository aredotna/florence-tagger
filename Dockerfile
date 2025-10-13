FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# sanity: show python/torch versions at build
RUN python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__)"

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR /app

# 1) flash-attn wheel for torch 2.2.2/cu121/py310
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.5.6

# 2) app deps (transformers, spacy, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) spaCy model
RUN python -m spacy download en_core_web_sm

# 4) app
COPY app.py ./

ENV AWS_REGION=us-east-1 \
    FLORENCE_MODEL_ID=microsoft/Florence-2-base \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
