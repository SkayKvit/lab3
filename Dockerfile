# =========================
# Stage 1: TRAINER
# =========================
FROM python:3.11-slim AS trainer

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps (audio libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torchcodec

# Copy training code
COPY train.py .

# Prepare dirs
RUN mkdir -p /artifacts /data/speech_commands

# Train model during build
RUN python train.py --save-model /artifacts/model.pth --download-data

# =========================
# Stage 2: RUNTIME
# =========================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install lightweight audio deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies (only the essentials)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install numpy python-multipart fastapi uvicorn torch torchaudio

# Copy trained model
COPY --from=trainer /artifacts/model.pth /app/artifacts/model.pth

# Copy inference app
COPY app.py .
COPY train.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--workers", "1"]
