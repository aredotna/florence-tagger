# Simple Python 3.10 image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app.py ./

ENV AWS_REGION=us-east-1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


