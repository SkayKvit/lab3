# =========================
# Stage 1: TRAINER
# =========================
FROM python:3.11-slim AS trainer

ENV DEBIAN_FRONTEND=noninteractive

# Системні залежності для аудіо та збірки
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Копіюємо requirements і встановлюємо всі пакети разом без кешу
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.2+cpu torchaudio==2.2.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torchcodec

# Копіюємо тренувальний код
COPY train.py .

# Папки для артефактів та даних
RUN mkdir -p /artifacts /data/speech_commands

# Тренуємо модель під час build
RUN python train.py --save-model /artifacts/model.pth --download-data


# =========================
# Stage 2: RUNTIME
# =========================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Мінімальні системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо requirements і встановлюємо тільки потрібне для inference
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        numpy python-multipart fastapi uvicorn \
        torch==2.2.2+cpu torchaudio==2.2.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Копіюємо натреновану модель з тренера
COPY --from=trainer /artifacts/model.pth /app/artifacts/model.pth

# Копіюємо inference app
COPY app.py .
COPY train.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--workers", "1"]
