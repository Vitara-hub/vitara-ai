# ── Stage 1: Builder ────────────────────────────────────────────────────────
# Gunakan base image Python yang kompatibel dengan TensorFlow dan Linux x86_64
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies yang dibutuhkan untuk build (gcc, g++ untuk ChromaDB/hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements lebih awal untuk memanfaatkan Docker layer cache
COPY requirements.txt .

# Install semua dependensi Python ke direktori /install
# Ini memisahkan dependencies dari source code agar build lebih efisien
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime system dependencies (libstdc++ untuk TensorFlow shared libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Salin installed packages dari stage builder
COPY --from=builder /install /usr/local

# Buat user non-root untuk keamanan (wajib di Hugging Face Spaces)
RUN useradd -m -u 1000 vitara

# Buat direktori untuk ChromaDB dan berikan hak akses /app ke user vitara
RUN mkdir -p data/chroma_db && chown -R vitara:vitara /app

USER vitara

# Copy seluruh source code aplikasi
COPY --chown=vitara:vitara . .

# ── Environment Defaults ──────────────────────────────────────────────────────
# APP_ENV diset ke production (menonaktifkan Swagger UI /docs dan /redoc)
# Ubah ke 'development' untuk mengaktifkan /docs saat debugging
ENV APP_ENV=production
ENV APP_PORT=7860
ENV CHROMA_DB_PATH=./data/chroma_db

# Expose port 7860 (port default Hugging Face Spaces)
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
