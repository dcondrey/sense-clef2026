# SENSE Docker Image for TIRA
# PAN@CLEF 2026 Sensemaking Task
# Supports both rubric (0-2) and simple (0-4) tracks
# Architecture: DeBERTa CORAL + LGB/MLP PoE Ensemble with Irish Translation

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[tira]"

# Pre-download models used at inference time
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/nli-deberta-v3-base')"

RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('cross-encoder/nli-deberta-v3-base'); \
    AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base'); \
    AutoModel.from_pretrained('microsoft/mdeberta-v3-base'); \
    AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base'); \
    AutoModel.from_pretrained('FacebookAI/xlm-roberta-large'); \
    AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large', use_fast=False)"

RUN python -c "from transformers import MarianMTModel, MarianTokenizer; \
    MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ga-en'); \
    MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ga-en')"

# Suppress HF Hub warnings at runtime (models are pre-downloaded above)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Copy application code
COPY sense/ sense/
COPY predict.py .

# Copy pre-trained models
# Run train.py BEFORE building Docker image
COPY models/tira/ models/tira/

ENTRYPOINT ["python", "predict.py"]
