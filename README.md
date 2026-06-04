# Vitara AI Project 🚀

Ini adalah _repository_ utama untuk tim AI dalam proyek **Vitara**. Proyek ini memuat model machine learning dan layanan inferensi (API) yang bertanggung jawab untuk fitur cerdas pada aplikasi Vitara.

## 📂 Struktur Repositori

Repositori ini menggunakan struktur modular untuk memisahkan antara riset (notebooks) dan produksi (service):

```text
.
├── vitara-ai-service/      # Aplikasi utama (FastAPI) untuk melayani inferensi model AI
├── notebooks/              # Jupyter Notebook untuk eksperimen, preprocessing, dan training model
├── tensor_scripts/         # Skrip utilitas untuk konversi model TensorFlow/Keras ke ONNX
├── data/                   # (Ignored) Folder lokal untuk raw data, dataset, atau model checkpoint
└── docs/                   # Dokumentasi arsitektur, API contract, dan postman collection
```

---

### 🧠 Modul AI (Microservices)

Berikut modul AI yang telah dikembangkan dan di-serve pada layanan API:

| Modul                       | Endpoint                | Status               | Keterangan                                                                                       |
| --------------------------- | ----------------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **NLP (Journal Analysis)**  | `POST /predict/journal` | ✅ Aktif             | Analisis emosi & tingkat stres dari teks jurnal menggunakan model IndoBERT (ONNX).               |
| **Vision (Food Detection)** | `POST /predict/food`    | ✅ Aktif             | Klasifikasi makanan & -estimasi kalori dari gambar (MobileNetV2 TFLite).                         |
| **Health Score**            | `POST /health/score`    | ✅ Aktif             | Kalkulasi skor kesehatan holistik berbasis **rule-based engine** (deterministic).                |
| **LLM Companion**           | `POST /companion/chat`  | ✅ Aktif             | Asisten kesehatan AI berbasis **Gemini 3.1 Flash (Lite) + RAG (ChromaDB)** dengan SSE streaming. |
| **Sleep Pattern**           | `POST /predict/sleep`   | ✅ Aktif (mock mode) | Analisis kualitas dan pola tidur pengguna (formula mock dinamis).                                |
| **Typing Pattern**          | `POST /predict/typing`  | ✅ Aktif             | Deteksi tingkat stres melalui dinamika pola pengetikan menggunakan model LSTM (ONNX).            |

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
   # Isi GEMINI_API_KEY di file .env untuk mengaktifkan fitur LLM Companion
   ```

4. **Jalankan API Server**:
   ```bash
   python main.py
   # atau
   uvicorn main:app --reload
   ```

Setelah server berjalan, dokumentasi interaktif tersedia di:
👉 **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
👉 **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧪 Testing & Verifikasi

Kami menyediakan beberapa metode pengujian:

- **Postman Collection**: Tersedia di `vitara-ai-service/docs/postman_collection.json` untuk pengujian integrasi otomatis. Koleksi ini mencakup validasi fungsionalitas dengan ambang batas **latency ≤ 2 detik** per endpoint.
- **Skrip LLM Companion**: `vitara-ai-service/scripts/test_companion.py` untuk menguji pipeline RAG (ChromaDB) dan streaming Gemini secara langsung dari terminal.
- **Validasi Model**: `vitara-ai-service/scripts/validate_models.py` untuk memvalidasi performa model AI terhadap ambang batas metrik yang telah ditentukan.

---

## 👥 Kontributor

- **Bagus** (AI Engineer)
- **Putri** (AI Engineer)
- **Reihan** (Data Scientist)
- **Hilmi** (Data Scientist)
