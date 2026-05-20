# Layanan AI Vitara (Vitara AI Service)

Ini adalah aplikasi **FastAPI** yang menjadi inti dari layanan AI Vitara — menangani inferensi model machine learning, integrasi LLM (Gemini), dan manajemen memori percakapan berbasis vektor (RAG).

---

## Ringkasan Arsitektur

`vitara-ai-service` menyediakan empat endpoint utama yang aktif saat ini, dengan dua modul lainnya masih dalam pengembangan:

| Journal Analysis | `POST /predict/journal` | NLP (TensorFlow `.keras`) | ✅ Aktif (mock mode jika model belum ada) |
| Food Detection | `POST /predict/food` | MobileNetV2 TFLite | ✅ Aktif (model diperlukan) |
| Health Score | `POST /health/score` | Rule-Based Engine | ✅ Aktif |
| LLM Companion | `POST /companion/chat` | Gemini 2.5 Flash + ChromaDB | ✅ Aktif |
| Sleep Analysis | `POST /predict/sleep` | Rule-Based / Mock | ✅ Aktif (mock mode) |
| Typing Analysis | `POST /predict/typing` | LSTM (TensorFlow `.h5`) | ✅ Aktif |


---

## Struktur Proyek

```text
vitara-ai-service/
├── main.py                 # Entry point FastAPI (menginisialisasi app & semua router)
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi kontainerisasi (Docker)
├── .env.example            # Template konfigurasi environment variable
│
├── routers/                # Definisi rute API (FastAPI Router)
│   ├── journal.py          # /predict/journal — NLP analisis emosi & stres
│   ├── food.py             # /predict/food   — Vision klasifikasi makanan
│   ├── health_score.py     # /health/score   — Kalkulasi skor kesehatan + RAG sync
│   ├── companion.py        # /companion/chat — LLM Companion SSE streaming
│   ├── sleep.py            # /predict/sleep  — Analisis tidur (mock)
│   └── typing.py           # /predict/typing — Analisis pola mengetik (LSTM)
│
├── services/               # Lapisan logika bisnis & layanan AI inti
│   ├── health_score_service.py  # Rule-based engine untuk kalkulasi health score
│   ├── llm_companion.py         # Pipeline RAG + Gemini streaming (LLM Companion)
│   ├── memory_store.py          # Abstraksi ChromaDB (add, retrieve, delete)
│   ├── context_builder.py       # Membangun konteks dari memori vektor untuk RAG
│   └── prompts.py               # System instruction & user prompt templates (Gemini)
│
├── schemas/                # Skema validasi data Pydantic
│   ├── journal.py          # JournalRequest, JournalResponse
│   ├── food.py             # FoodResponse
│   ├── health_score.py     # HealthScoreRequest, HealthScoreResponse, Breakdown
│   ├── companion.py        # CompanionChatRequest
│   ├── sleep.py            # SleepPredictRequest, SleepPredictResponse
│   └── typing.py           # TypingPredictRequest, TypingPredictResponse
│
├── models/                 # Model machine learning (file .keras/.tflite)
│   ├── nlp_model/          # Model NLP (nlp_model.keras)
│   └── vision_model/       # Model Vision TFLite + classes.txt
│
├── scripts/                # Skrip utilitas & pengujian
│   ├── test_companion.py        # Uji integrasi LLM Companion & ChromaDB
│   ├── clear_memories.py        # Reset/bersihkan data memori ChromaDB
│   └── validate_models.py       # Validasi performa semua model AI
│
├── docs/                   # Dokumentasi teknis
│   ├── architecture.md          # Arsitektur sistem Vitara AI
│   ├── api-contract.md          # Kontrak API lengkap
│   ├── dataset-contract.md      # Format & spesifikasi dataset
│   ├── inference-guide.md       # Panduan CLI inference standalone
│   ├── model-validation-guide.md
│   └── postman_collection.json  # Koleksi Postman untuk pengujian integrasi
│
├── data/                   # (Ignored) Data lokal & penyimpanan ChromaDB
│   └── chroma_db/          # Database vektor ChromaDB (persistent)
│
├── sample/                 # Contoh data untuk pengujian inferensi
└── logs/                   # Log aplikasi & TensorBoard
```

---

## Prasyarat

- **Python 3.9+**
- **uv** (direkomendasikan): Package manager Python yang cepat.
  - Instalasi: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - **Penting untuk macOS Apple Silicon**: Pastikan `uv` terinstal sebagai native ARM64. Jika mengalami error "AVX instructions", instal ulang dengan perintah di atas.
- **GEMINI_API_KEY** (diperlukan untuk fitur LLM Companion): Dapatkan dari [Google AI Studio](https://aistudio.google.com/).

---

## Instalasi & Setup

### 1. Menggunakan `uv` (Direkomendasikan)

```bash
# Pastikan PATH sudah terupdate (untuk macOS)
export PATH="$HOME/.local/bin:$PATH"

# Buat virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux

# Install dependensi
uv pip install -r requirements.txt
```

### 2. Menggunakan `pip` Standar

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Konfigurasi Environment Variable

```bash
cp .env.example .env
```

Edit file `.env` sesuai kebutuhan:

```env
# FastAPI Configuration
APP_ENV=development
APP_PORT=8000

# LLM Companion Configuration (wajib untuk fitur Companion)
GEMINI_API_KEY=your_gemini_api_key_here

# Vector Database (RAG)
CHROMA_DB_PATH=./data/chroma_db
```

---

## Menjalankan API Service

```bash
# Menggunakan uvicorn secara langsung (dengan hot-reload)
uvicorn main:app --reload

# Menggunakan uv run (direkomendasikan)
uv run uvicorn main:app --reload

# Atau menjalankan main.py
python main.py
```

Setelah server berjalan, akses dokumentasi interaktif di:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Endpoint API

### 📰 Journal Analysis
**`POST /predict/journal`** — Menganalisis teks jurnal untuk mendeteksi emosi, tingkat stres, dan topik.

- **Request**: `{ "text": "...", "user_id": "..." }`
- **Response**: `{ "emotion": "anxious", "stress_level": 0.82, "topics": ["deadline", "kerja"] }`
- **Catatan**: Otomatis menyimpan hasil analisis ke memori RAG (ChromaDB).

### 🍔 Food Detection
**`POST /predict/food`** — Mengklasifikasikan makanan dan mengestimasi kalori dari gambar.

- **Request**: `multipart/form-data` — field `image` (JPEG/PNG) & `user_id` (opsional)
- **Response**: `{ "foods": ["nasi_goreng"], "estimated_calories": 450 }`
- **Catatan**: Memerlukan model `vision_model.tflite` di folder `models/vision_model/`.

### 💚 Health Score
**`POST /health/score`** — Menghitung skor kesehatan holistik secara **rule-based & deterministic**.

- **Request**: Mendukung **input parsial** — cukup kirimkan data yang tersedia.
  ```json
  {
    "user_id": "user-123",
    "nlp_result": { "emotion": "anxious", "stress_level": 0.82 },
    "food_result": { "estimated_calories": 450 },
    "sleep_result": { "quality_score": 75 },
    "typing_result": { "stress_score": 0.6 }
  }
  ```
- **Response**: `{ "health_score": 72, "breakdown": { "mood": 40, "nutrition": 90, "stress": 41, "sleep": 75 } }`
- **Catatan**: Hasil perhitungan otomatis disinkronkan ke ChromaDB secara **asinkron** (background task) sebagai konteks untuk LLM Companion.

### 🤖 LLM Companion (Gemini RAG)
**`POST /companion/chat`** — Asisten kesehatan AI yang berkomunikasi via **Server-Sent Events (SSE)** streaming.

- **Request**: `{ "user_id": "user-123", "message": "Aku merasa kelelahan hari ini." }`
- **Response**: Stream SSE dengan dua jenis event:
  - `event: delta` — token teks demi token secara real-time.
  - `event: final` — respons lengkap beserta 2-4 rekomendasi kesehatan terstruktur.
- **Pipeline RAG**: Setiap permintaan secara otomatis mengambil memori relevan dari ChromaDB → membangun konteks → mengirim ke Gemini 2.5 Flash → menyimpan kembali interaksi ke ChromaDB.
- **Catatan**: Memerlukan `GEMINI_API_KEY` di file `.env`. Jika tidak dikonfigurasi, berjalan dalam **Demo Mode**.

### 💤 Sleep Analysis
**`POST /predict/sleep`** — Menghitung skor kualitas tidur pengguna berdasarkan log tidur.

- **Request**:
  ```json
  {
    "duration_hours": 5.5,
    "bedtime": "00:30",
    "wake_time": "06:00",
    "interruptions": 3,
    "sleep_debt_hours": 2.0,
    "user_id": "usr_abc123"
  }
  ```
- **Response**: `{ "quality_score": 72 }`
- **Catatan**: Menggunakan formula mock dinamis untuk saat ini. Otomatis menyimpan hasil analisis ke RAG jika `user_id` disertakan.

### ⌨️ Typing Analysis
**`POST /predict/typing`** — Mendeteksi tingkat stres berdasarkan pola pengetikan (keystroke dynamics).

- **Request**:
  ```json
  {
    "wpm": 58.3,
    "backspace_rate": 0.12,
    "inter_key_timings": [120, 98, 145, 87, 203, 110],
    "user_id": "usr_abc123"
  }
  ```
- **Response**: `{ "stress_score": 0.74 }`
- **Catatan**: Menggunakan model LSTM (`typing_stress_lstm.h5`). Otomatis menyimpan hasil analisis ke RAG jika `user_id` disertakan.

---


## Inference Standalone (CLI)

Untuk pengujian model secara mandiri tanpa menjalankan server web penuh, tersedia skrip inference standalone:

```bash
# Food Vision
python inference_food.py --image sample/food.jpg

# NLP
python inference_nlp.py

# Sleep
python inference_sleep.py

# Typing
python inference_typing.py
```

Panduan lengkap parameter CLI tersedia di 👉 **[docs/inference-guide.md](docs/inference-guide.md)**

---

## Testing & Integrasi

### 1. Postman Collection

Import `docs/postman_collection.json` ke Postman, lalu jalankan dengan server yang sudah berjalan di `localhost:8000`. Koleksi sudah dilengkapi **Test Scripts** untuk memvalidasi respons dan latency ≤ 2 detik.

### 2. Skrip Uji Integrasi LLM Companion

```bash
# Menggunakan uv (direkomendasikan):
uv run python scripts/test_companion.py

# Menggunakan python standar (pastikan venv aktif):
python scripts/test_companion.py
```

**Yang diuji:**
- ✅ **Memory Store (ChromaDB)**: Upsert memori, semantic retrieval, dan cleanup data uji.
- ✅ **LLM Companion**: Deteksi `GEMINI_API_KEY`, visualisasi streaming SSE token demi token, validasi data rekomendasi terstruktur.

### 3. Skrip Pembersihan ChromaDB

```bash
# Menghapus semua data memori pengguna dari ChromaDB:
uv run python scripts/clear_memories.py

# Hard reset total (hapus folder fisik database):
rm -rf data/chroma_db
```

### 4. Validasi Model

```bash
# Validasi semua model sekaligus:
uv run python scripts/validate_models.py --all

# Validasi per model:
uv run python scripts/validate_models.py --nlp
uv run python scripts/validate_models.py --vision
uv run python scripts/validate_models.py --typing
uv run python scripts/validate_models.py --sleep
uv run python scripts/validate_models.py --health
```

Skrip akan mengeluarkan status `PASS` atau `FAIL` per metrik, dan mengembalikan exit code `1` jika ada model yang tidak memenuhi ambang batas.

---

## Monitoring dengan TensorBoard

```bash
# Menggunakan uv:
uv run tensorboard --logdir logs/

# Menggunakan python standar:
tensorboard --logdir logs/
```

Akses di browser: **[http://localhost:6006](http://localhost:6006)**

| Tab | Fungsi |
|---|---|
| **Scalars** | Pantau grafik `accuracy` & `loss`. Waspadai overfitting (training loss ↓, validation loss ↑). |
| **Graphs** | Periksa struktur arsitektur model secara visual. |
| **Histograms** | Lihat distribusi _weights_ & _bias_ selama pelatihan. |
