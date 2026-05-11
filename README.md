# Vitara AI Project 🚀

Ini adalah _repository_ utama untuk tim AI dalam proyek **Vitara**. Proyek ini memuat model machine learning dan layanan inferensi (API) yang bertanggung jawab untuk fitur cerdas pada aplikasi Vitara.

## 📂 Struktur Repositori

Repositori ini menggunakan struktur modular untuk memisahkan antara riset (notebooks) dan produksi (service):

```text
.
├── vitara-ai-service/      # Aplikasi utama (FastAPI) untuk melayani inferensi model AI
├── notebooks/              # Jupyter Notebook untuk eksperimen, preprocessing, dan training model
├── data/                   # (Ignored) Folder lokal untuk raw data, dataset, atau model checkpoint
└── docs/                   # Dokumentasi arsitektur, API contract, dan postman collection
```

### 🧠 Modul AI (Microservices)

Beberapa model yang dikembangkan dan di-serve pada servis API ini di antaranya:

- **NLP (Journal Analysis)**: Analisis emosi dan tingkat stres berbasis teks jurnal.
- **Vision (Food Detection)**: Pengenalan jenis makanan dan estimasi kalori berbasis citra.
- **Health Score**: Model multimodal untuk menghitung skor kesehatan holistik.
- **Sleep Pattern**: Analisis kualitas dan pola tidur pengguna.
- **Typing Pattern**: Deteksi tingkat stres melalui dinamika pola pengetikan.

---

## 🚀 Cara Menjalankan (Local Development)

Layanan API berada di dalam folder `vitara-ai-service`. Ikuti langkah berikut untuk menjalankan secara lokal:

1. **Siapkan Environment** (Direkomendasikan menggunakan `uv` untuk performa lebih cepat):

   ```bash
   cd vitara-ai-service
   uv venv
   source .venv/bin/activate
   ```

2. **Install Dependensi**:

   ```bash
   uv pip install -r requirements.txt
   ```

3. **Konfigurasi Environment Variable**:

   ```bash
   cp .env.example .env
   ```

4. **Jalankan API Server**:
   ```bash
   python main.py
   # atau
   uvicorn main:app --reload
   ```

Setelah server berjalan, dokumentasi interaktif tersedia di:
👉 **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing & Verifikasi

Kami menggunakan Postman untuk pengujian integrasi otomatis. Koleksi pengujian tersedia di:
`vitara-ai-service/docs/postman_collection.json`

Koleksi ini mencakup pengujian fungsionalitas dan validasi performa dengan ambang batas **latency ≤ 2 detik** per endpoint.

---

## 👥 Kontributor

- **Bagus** (AI Engineer)
- **Putri** (AI Engineer)
- **Reihan** (Data Scientist)
- **Hilmi** (Data Scientist)
