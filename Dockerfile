FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install requirements first
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy English model properly (this is the right command)
RUN python -m spacy download en_core_web_sm

COPY app.py ./

EXPOSE 8000

ENV AWS_REGION=us-east-1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
