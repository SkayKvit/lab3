# =========================
# Stage 1: TRAINER
# =========================
FROM python:3.11-slim AS trainer

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Мінімальні системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg libsndfile1 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
# Встановлюємо спочатку великі пакети
RUN pip install --no-cache-dir torch==2.2.2+cpu torchaudio==2.2.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
# Потім решту залежностей
RUN pip install --no-cache-dir -r requirements.txt

# Якщо модель вже є, копіюємо її
COPY model.pth /artifacts/model.pth

# =========================
# Stage 2: RUNTIME
# =========================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Мінімальні системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for inference
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy python-multipart fastapi uvicorn
RUN pip install --no-cache-dir torch==2.2.2+cpu torchaudio==2.2.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy pretrained model from trainer stage
COPY --from=trainer /artifacts/model.pth /app/artifacts/model.pth

# Copy inference app
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--workers", "1"]
