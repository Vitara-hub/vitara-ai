# Layanan AI Vitara (Vitara AI Service)

Ini adalah aplikasi **FastAPI** yang menjadi inti dari layanan AI Vitara — menangani inferensi model machine learning, integrasi LLM (Gemini), dan manajemen memori percakapan berbasis vektor (RAG).

---

## Ringkasan Arsitektur

`vitara-ai-service` menyediakan empat endpoint utama yang aktif saat ini, dengan dua modul lainnya masih dalam pengembangan:

| Journal Analysis | `POST /predict/journal` | NLP (ONNX + Tokenizer Lokal) | ✅ Aktif |
| Food Detection | `POST /predict/food` | MobileNetV2 TFLite | ✅ Aktif (model diperlukan) |
| Health Score | `POST /health/score` | Rule-Based Engine | ✅ Aktif |
| LLM Companion | `POST /companion/chat` | Gemini 3.1 Flash (Lite) + ChromaDB | ✅ Aktif |
| Sleep Analysis | `POST /predict/sleep` | Rule-Based / Mock | ✅ Aktif (mock mode) |
| Typing Analysis | `POST /predict/typing` | LSTM ONNX | ✅ Aktif |


---

## Struktur Proyek

```text
vitara-ai-service/
├── main.py                 # Entry point FastAPI (menginisialisasi app & semua router)
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi kontainerisasi (Docker)
├── .env.example            # Template konfigurasi environment variable
├── README.md               # Dokumentasi utama (file ini)
├── README_HF.md            # Dokumentasi Hugging Face Spaces
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
├── models/                 # Model machine learning (file .onnx/.tflite)
│   ├── NLP_Rai/            # Direktori eksperimen NLP
│   ├── health_score_model/ # Model/Data Health Score
│   ├── nlp_model/          # Model NLP (vitara_nlp_indobert.onnx & folder tokenizer/)
│   ├── sleep_model/        # Model Analisis Tidur
│   ├── typing_model/       # Model Typing (typing_stress_lstm.onnx)
│   ├── vision_model/       # Model Vision TFLite + classes.txt
│   ├── custom_callbacks.py # Custom callbacks untuk TensorFlow/Keras
│   ├── custom_layers.py    # Custom layers untuk TensorFlow/Keras
│   └── custom_losses.py    # Custom losses untuk TensorFlow/Keras
│
├── scripts/                # Skrip utilitas & pengujian
│   ├── check_memory.py          # Cek isi memori ChromaDB per user
│   ├── clear_memories.py        # Reset/bersihkan data memori ChromaDB
│   ├── deploy_hf.sh             # Skrip deploy otomatis ke Hugging Face Spaces
│   ├── test_companion.py        # Uji integrasi LLM Companion & ChromaDB
│   └── validate_models.py       # Validasi performa semua model AI
│
├── docs/                   # Dokumentasi teknis
│   ├── Vitara - Your Whole Health, One Place.md # Detail produk Vitara
│   ├── api-contract.md          # Kontrak API lengkap
│   ├── architecture.md          # Arsitektur sistem Vitara AI
│   ├── colab_custom_callbacks_guide.md # Panduan custom callbacks Colab
│   ├── dataset-contract.md      # Format & spesifikasi dataset
│   ├── inference-guide.md       # Panduan CLI inference standalone
│   ├── model-validation-guide.md
│   └── postman_collection.json  # Koleksi Postman untuk pengujian integrasi
│
├── data/                   # (Ignored) Data lokal & penyimpanan ChromaDB
│   └── chroma_db/          # Database vektor ChromaDB (persistent)
│
├── sample/                 # Contoh data untuk pengujian inferensi
├── logs/                   # Log aplikasi & TensorBoard
├── inference_food.py       # Skrip CLI untuk pengujian inference food
├── inference_nlp.py        # Skrip CLI untuk pengujian inference NLP
├── inference_sleep.py      # Skrip CLI untuk pengujian inference sleep
└── inference_typing.py     # Skrip CLI untuk pengujian inference typing
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
COMPANION_MODEL_NAME=gemini-3.1-flash-lite  # Opsional, default: gemini-3.1-flash-lite

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
- **Pipeline RAG**: Setiap permintaan secara otomatis mengambil memori relevan dari ChromaDB → membangun konteks → mengirim ke Gemini 3.1 Flash (Lite) (atau model yang dikonfigurasi) → menyimpan kembali interaksi ke ChromaDB.
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
- **Catatan**: Menggunakan formula mock dinamis untuk saat ini. Otomatis menyimpan hasil analisis ke RAG jika `user_id` disertakan. Menggunakan **daily upsert** — jika dipanggil beberapa kali dalam sehari, hanya data tidur terbaru yang disimpan.

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
- **Catatan**: Menggunakan model LSTM dalam format ONNX (`typing_stress_lstm.onnx`). Otomatis menyimpan hasil analisis ke RAG jika `user_id` disertakan. Menggunakan **daily upsert** — jika dipanggil beberapa kali dalam sehari, hanya data sesi terbaru yang disimpan.

---

## Sistem Memori RAG (ChromaDB)

Setiap endpoint yang menghasilkan data kesehatan secara otomatis menyimpan hasilnya ke **ChromaDB** (vector database persisten). Sistem ini digunakan sebagai konteks oleh **LLM Companion** untuk memberikan respons yang personal dan berbasis riwayat kesehatan pengguna.

### Strategi Penyimpanan

Ada dua strategi penyimpanan yang diterapkan berdasarkan karakteristik masing-masing data:

#### 🔄 Daily Upsert — Overwrite per Hari
Digunakan untuk data yang hanya memiliki satu nilai relevan per hari. Jika endpoint dipanggil lagi pada hari yang sama, entry lama **diganti** dengan data terbaru. Menggunakan `doc_id` deterministik: `{tipe}_{user_id}_{YYYY-MM-DD}`.

| Tipe Memori (`mem_type`) | Router | Keterangan |
|---|---|---|
| `sleep_prediction` | `/predict/sleep` | Hanya ada satu siklus tidur per malam |
| `typing_prediction` | `/predict/typing` | Kondisi stres pengetikan terkini hari ini |
| `health_score` | `/health/score` | Skor kesehatan harian (recalculated) |

#### 📚 Keep ALL — Akumulasi
Digunakan untuk data yang setiap entry-nya memiliki nilai unik dan informatif meskipun dikirim berkali-kali. Menggunakan `doc_id` random (UUID) setiap kali.

| Tipe Memori (`mem_type`) | Router | Keterangan |
|---|---|---|
| `user_journal` | `/predict/journal` | Teks asli jurnal — setiap jurnal adalah narasi emosi yang unik |
| `nlp_prediction` | `/predict/journal` | Hasil analisis emosi & stres dari setiap jurnal |
| `food_prediction` | `/predict/food` | Setiap makanan berbeda, user bisa makan 3x/hari |
| `user_chat` | `/companion/chat` | Setiap pesan percakapan adalah riwayat unik |
| `companion_response` | `/companion/chat` | Setiap respons companion adalah riwayat unik |

### Pengambilan Memori (Diversified Retrieval)

LLM Companion menggunakan **`retrieve_diverse_memories()`** — bukan flat top-k — untuk memastikan setiap tipe memori selalu terwakili dalam konteks yang dikirim ke Gemini.

```
Untuk setiap mem_type → ambil top-2 paling semantically relevan
                       → gabungkan & urutkan kronologis
                       → kirim ke Gemini sebagai konteks RAG
```

Dengan strategi ini, konteks yang dibangun selalu mencakup riwayat dari **semua dimensi kesehatan** (jurnal, makanan, tidur, pengetikan, health score) secara proporsional, bukan didominasi oleh tipe yang paling sering dikirim.

### Script Debug Memory

```bash
# Lihat semua memori tersimpan untuk user tertentu, dikelompokkan per tipe:
uv run python scripts/check_memory.py <user_id>
```

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

> [!IMPORTANT]
> Perintah `uv run` untuk TensorBoard harus dijalankan dari folder `tensor_scripts/` karena dependensi TensorBoard terpasang di virtual environment folder tersebut.

```bash
# Menggunakan uv (jalankan dari folder tensor_scripts/):
cd ../tensor_scripts
uv run tensorboard --logdir ../vitara-ai-service/logs/

# Menggunakan python standar (dari folder vitara-ai-service/):
tensorboard --logdir logs/
```

Akses di browser: **[http://localhost:6006](http://localhost:6006)**

| Tab | Fungsi |
|---|---|
| **Scalars** | Pantau grafik `accuracy` & `loss`. Waspadai overfitting (training loss ↓, validation loss ↑). |
| **Graphs** | Periksa struktur arsitektur model secara visual. |
| **Histograms** | Lihat distribusi _weights_ & _bias_ selama pelatihan. |

---

## Deployment (Hugging Face Spaces)

Layanan ini dikonfigurasi untuk dapat di-deploy secara gratis ke **Hugging Face Spaces** menggunakan **Docker**.

### Prasyarat Deployment
1. Buat Space baru di Hugging Face dengan nama `vitara-ai-service`.
2. Pilih SDK **Docker** (pilih Blank template).
3. Buat **Access Token (Write)** di Hugging Face (Settings -> Access Tokens).
4. Lakukan clone repositori Space tersebut di luar folder `vitara-ai` Anda:
   ```bash
   cd "/path/to/parent-folder"
   git clone https://huggingface.co/spaces/bagususwanto/vitara-ai-service
   ```

### Cara Deploy & Update Otomatis
Gunakan skrip otomatisasi [deploy_hf.sh](scripts/deploy_hf.sh) yang berada di dalam folder `scripts/` untuk menyalin perubahan, mengemas, dan mengunggahnya ke Hugging Face:

```bash
# 1. Masuk ke folder service utama Anda
cd "/path/to/parent-folder/vitara-ai/vitara-ai-service"

# 2. Berikan izin eksekusi pada skrip (cukup sekali saja)
chmod +x scripts/deploy_hf.sh

# 3. Jalankan skrip deploy dengan pesan commit kustom Anda
./scripts/deploy_hf.sh "Menambahkan fitur baru atau update model"
```

### Konfigurasi Environment Secrets
Pastikan Anda menambahkan Environment Secrets berikut di menu **Settings -> Variables and secrets** pada halaman Space Anda:
- `GEMINI_API_KEY`: API Key dari Google AI Studio untuk LLM Companion.
