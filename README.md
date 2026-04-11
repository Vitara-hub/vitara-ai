# Vitara AI Project

Ini adalah _repository_ utama untuk tim AI dalam proyek Vitara. Proyek ini memuat model dan layanan inferensi (API) yang bertanggung jawab untuk fitur cerdas pada aplikasi Vitara.

## 📂 Struktur Repositori

Secara garis besar, repositori ini dibagi menjadi beberapa bagian utama:

```text
.
├── vitara-ai-service/      # Aplikasi utama (FastAPI) untuk melayani inferensi model AI
├── notebooks/              # Jupyter Notebook yang digunakan untuk eksperimen, preprocessing, dan training model
└── data/                   # (Tidak di-commit) Folder lokal untuk raw data, dataset, atau model checkpoint sementara
```

### 🧠 Modul AI (Microservices)

Beberapa model yang dikembangkan dan di-serve pada servis API ini di antaranya:

- **NLP (Journal Analysis)**: Model analisis sentimen/emosi berbasis jurnal teks pengguna.
- **Vision (Food Detection)**: Model pengenalan makanan berbasis citra (gambar).
- **Health Score**: Model multimodal untuk menghitung skor kesehatan berdasarkan input user.
- **Sleep Pattern**: Model yang menganalisis dan memprediksi kualitas pola tidur.
- **Typing Pattern**: Model yang mengkaji pola pengetikan user.

## 🚀 Cara Menjalankan (Local Development)

_(Dokumentasi lebih lanjut ini akan terus di-update ke depannya)_

1. Masuk ke environment virtual pilihan (misalnya `venv` atau `conda`).
2. Install dependensi (bila ada, jalankan pada folder `vitara-ai-service`):
   ```bash
   cd vitara-ai-service
   pip install -r requirements.txt
   ```
3. Copy isi dari `.env.example` ke dalam file `.env`:
   ```bash
   cp .env.example .env
   ```
4. Jalankan _server_ development menggunakan `uvicorn`:
   ```bash
   uvicorn main:app --reload
   ```
