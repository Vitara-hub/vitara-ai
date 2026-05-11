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
- `main.py`: Titik masuk (_entry point_) utama untuk layanan API (saat ini sedang dalam pengembangan).
- `requirements.txt`: Dependensi proyek Python.
- `Dockerfile`: Konfigurasi kontainerisasi (Docker).

## Pengaturan & Instalasi

### 1. Prasyarat

Pastikan Python 3.9+ telah terinstal di sistem Anda. Kami sangat merekomendasikan penggunaan **[uv](https://github.com/astral-sh/uv)** untuk manajemen paket yang jauh lebih cepat.

### 2. Instalasi (Menggunakan `uv` - Direkomendasikan)

```bash
# Membuat virtual environment
uv venv

# Mengaktifkan virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

```bash
# Instalasi dependensi
uv pip install -r requirements.txt
```

### 3. Instalasi Standar (Tanpa `uv`)

Jika Anda tidak menggunakan `uv`, gunakan perintah standar berikut:

```bash
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows
pip install -r requirements.txt
```

## Contoh Penggunaan

Untuk menjalankan skrip inferensi makanan (_food vision_) secara langsung melalui terminal (_command line_), gunakan format berikut:

**Format Perintah:**

```bash
# Jika virtual environment sudah aktif:
python inference_food.py <path_ke_gambar> <path_ke_model_tflite> [path_ke_file_classes_txt]

# Atau menggunakan uv run (otomatis menggunakan venv):
uv run python inference_food.py <path_ke_gambar> <path_ke_model_tflite> [path_ke_file_classes_txt]
```

**Keterangan Parameter:**

- `<path_ke_gambar>`: Lokasi file gambar makanan yang ingin diprediksi (contoh: `sample_food.jpg`).
- `<path_ke_model_tflite>`: Lokasi model berformat `.tflite` yang sudah dilatih (contoh: `models/food_vision.tflite`).
- `[path_ke_file_classes_txt]`: _(Opsional)_ Lokasi file teks berisi daftar label kelas makanan.

**Contoh Menjalankan Skrip:**

```bash
python inference_food.py ./sample_food.jpg ./models/vision_model/vision_model.tflite ./models/vision_model/classes.txt
```

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
python scripts/validate_models.py --all
```

**2. Validasi Model Spesifik:**

- **NLP:** `python scripts/validate_models.py --nlp`
- **Vision:** `python scripts/validate_models.py --vision`
- **Secondary (Typing & Sleep):** `python scripts/validate_models.py --secondary`
- **Health Score:** `python scripts/validate_models.py --health`

### Hasil Validasi

Skrip akan memberikan output berupa status `PASS` atau `FAIL` untuk setiap metrik. Jika ada model yang tidak memenuhi ambang batas, skrip akan mengembalikan kode keluar (_exit code_) 1.

---

_Catatan: Ini adalah README sementara dan akan terus diperbarui seiring dengan pengembangan dan penerapan layanan API ini._
