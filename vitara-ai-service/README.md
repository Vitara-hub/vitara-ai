# Layanan AI Vitara (Vitara AI Service)

Repositori ini berisi layanan AI dan skrip _inference_ (inferensi) untuk proyek Vitara AI.

## Ringkasan

`vitara-ai-service` bertanggung jawab untuk menangani inferensi model AI di berbagai domain dalam aplikasi Vitara. Saat ini, layanan ini menyediakan skrip inferensi mandiri (_standalone_) yang dapat dipanggil untuk memproses data dan mengembalikan prediksi.

### Modul Inferensi

- **Food Vision (`inference_food.py`)**: Menggunakan model multi-output MobileNetV2 TFLite untuk mengklasifikasikan jenis makanan dan memperkirakan kandungan kalori dari gambar.
- **NLP (`inference_nlp.py`)**: Skrip inferensi untuk Pemrosesan Bahasa Alami (Natural Language Processing).
- **Sleep (`inference_sleep.py`)**: Skrip inferensi untuk analisis data terkait tidur.
- **Typing (`inference_typing.py`)**: Skrip inferensi untuk analisis perilaku mengetik.

## Struktur Proyek

- `models/`: Berisi model _machine learning_ yang telah dilatih (misalnya, model `.tflite`).
- `services/`: Lapisan logika bisnis dan layanan AI inti.
- `routers/`: Definisi rute API (untuk integrasi FastAPI).
- `sample/`: Contoh data untuk pengujian inferensi.
- `schemas/`: Skema validasi data serta _request/response_ (misalnya, menggunakan Pydantic).
- `scripts/`: Skrip utilitas untuk pemrosesan data, pelatihan, atau evaluasi.
- `docs/`: Dokumentasi tambahan.
- `logs/`: Log aplikasi.
- `main.py`: Titik masuk (_entry point_) utama untuk layanan API FastAPI.
- `requirements.txt`: Dependensi proyek Python.
- `Dockerfile`: Konfigurasi kontainerisasi (Docker).

## Menjalankan API Service

Layanan API menggunakan **FastAPI** dan dapat dijalankan dengan **Uvicorn**.

### 1. Menjalankan secara Lokal
Pastikan virtual environment Anda sudah aktif, lalu jalankan:

```bash
# Menggunakan uvicorn secara langsung
uvicorn main:app --reload

# Atau menjalankan main.py
python main.py
```

Setelah server berjalan, Anda dapat mengakses dokumentasi interaktif di:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Endpoints API (Batch 1)

| Endpoint | Method | Deskripsi |
| --- | --- | --- |
| `/predict/journal` | POST | Menganalisis teks jurnal untuk mendeteksi emosi dan tingkat stres. |
| `/predict/food` | POST | Mengklasifikasikan makanan dan mengestimasi kalori dari gambar. |

---

## Testing & Integrasi

### Postman Collection
Terdapat file koleksi Postman untuk mempermudah pengujian integrasi di:
`docs/postman_collection.json`

**Cara menggunakan:**
1. Import file `docs/postman_collection.json` ke dalam aplikasi Postman.
2. Pastikan server FastAPI sudah berjalan di `localhost:8000`.
3. Jalankan request yang tersedia. Koleksi ini sudah dilengkapi dengan **Test Scripts** untuk memvalidasi respons dan memastikan **latency ≤ 2 detik**.

---

## Contoh Penggunaan CLI (Inference Standalone)

### 1. Prasyarat

- **Python 3.9+**: Pastikan Python telah terinstal.
- **uv (Direkomendasikan)**: Kami sangat merekomendasikan penggunaan **[uv](https://github.com/astral-sh/uv)** untuk manajemen paket yang cepat.
  - **Penting untuk pengguna macOS (Apple Silicon)**: Pastikan `uv` diinstal sebagai native ARM64. Jika Anda mengalami error terkait "AVX instructions", instal ulang `uv` dengan perintah:
    `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Instalasi (Menggunakan `uv`)

```bash
# Untuk pengguna macOS, pastikan PATH sudah terupdate
export PATH="$HOME/.local/bin:$PATH"

# Membuat virtual environment
uv venv

# Instalasi dependensi
# (Otomatis mendeteksi platform: macOS Apple Silicon vs Windows/Linux)
uv pip install -r requirements.txt
```

### 3. Instalasi Standar (Tanpa `uv`)

Jika menggunakan `pip` standar, pastikan virtual environment Anda aktif:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## Contoh Penggunaan CLI (Inference Standalone)

Untuk memudahkan pengujian dan verifikasi model mandiri secara langsung melalui terminal (*command line*) tanpa menjalankan server web secara penuh, kami menyediakan skrip inferensi mandiri untuk masing-masing model (seperti Food Vision dan Typing Stress).

Panduan lengkap mengenai parameter input, format perintah, dan contoh penggunaan CLI untuk setiap model dapat diakses di:
👉 **[Inference Standalone CLI Guide](docs/inference-guide.md)**

---



## Validasi Model

Skrip `scripts/validate_models.py` digunakan untuk memvalidasi performa model AI (NLP, Vision, Sleep, Typing, dan Health Score) terhadap dataset pengujian (_test dataset_). Skrip ini membandingkan metrik performa aktual (seperti Akurasi atau MAE) dengan ambang batas (_threshold_) yang telah ditentukan.

### Persyaratan Data

Skrip ini mengasumsikan struktur data berikut di direktori root proyek:

- `data/nlp/processed/test.csv`
- `data/vision/processed/split/test/`
- `data/typing/processed/test.csv`
- `data/sleep/processed/test.csv`
- `data/health_score/processed/test.csv`

### Cara Menjalankan Validasi

Anda dapat menjalankan validasi untuk semua model sekaligus atau untuk model tertentu saja.

**1. Validasi Semua Model:**

```bash
# Menggunakan uv (direkomendasikan):
uv run python scripts/validate_models.py --all

# Menggunakan python standar (pastikan venv aktif):
python scripts/validate_models.py --all
```

**2. Validasi Model Spesifik:**

- **NLP:** `uv run python scripts/validate_models.py --nlp`
- **Vision:** `uv run python scripts/validate_models.py --vision`
- **Secondary (Typing & Sleep):** `uv run python scripts/validate_models.py --secondary`
- **Health Score:** `uv run python scripts/validate_models.py --health`

### Hasil Validasi

Skrip akan memberikan output berupa status `PASS` atau `FAIL` untuk setiap metrik. Jika ada model yang tidak memenuhi ambang batas, skrip akan mengembalikan kode keluar (_exit code_) 1.

---

## Monitoring dengan TensorBoard

TensorBoard digunakan untuk memvisualisasikan metrik pelatihan model (loss, accuracy, dll.) yang tersimpan di direktori `logs/`.

### Cara Menjalankan TensorBoard
Jalankan perintah berikut di direktori `vitara-ai-service`:

```bash
# Menggunakan uv (direkomendasikan):
uv run tensorboard --logdir logs/

# Menggunakan python standar:
tensorboard --logdir logs/
```

Setelah dijalankan, buka browser dan akses: **[http://localhost:6006](http://localhost:6006)**

### Hal yang Perlu Di-review
1. **Scalars Tab**: Perhatikan grafik `accuracy` dan `loss`.
   - Bandingkan garis *Training* dan *Validation*.
   - Waspadai **Overfitting**: Jika *training loss* terus turun tetapi *validation loss* justru naik.
2. **Graphs Tab**: Untuk memeriksa struktur arsitektur model secara visual.
3. **Histograms**: Untuk melihat distribusi bobot (_weights_) dan bias selama pelatihan.

---

_Catatan: Ini adalah README sementara dan akan terus diperbarui seiring dengan pengembangan dan penerapan layanan API ini._
