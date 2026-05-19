# 🧪 Model Validation Guide

Dokumen ini menjelaskan cara menggunakan script `scripts/validate_models.py` untuk memverifikasi performa model AI di Vitara AI Service sebelum diintegrasikan ke API endpoint.

## 📌 Ringkasan

Script validasi ini berfungsi sebagai _gatekeeper_ untuk memastikan semua model (NLP, Vision, Typing, Sleep, Health Score) telah memenuhi target metrik yang ditetapkan dalam project plan.

## 🛠️ Persiapan

### 1. Dependensi

Pastikan environment Anda telah menginstal package berikut:

```bash
pip install tensorflow pandas numpy
```

### 2. Struktur Folder

Script mengharapkan struktur folder sebagai berikut:

- `models/`: Berisi direktori model (SavedModel format).
- `data/`: Berisi dataset hasil preprocessing (`test.csv`, direktori `X_seq.npy`, `X_static.npy`, dan `y.npy`, atau direktori gambar).

### 3. Custom Components

Pastikan Putri telah memindahkan komponen custom dari Colab ke file lokal:

- `models/custom_layers.py` (e.g., `AttentionLayer`)
- `models/custom_losses.py` (e.g., `WeightedFocalLoss`)

## 🚀 Cara Penggunaan

Jalankan script dari root direktori `vitara-ai-service/`:

### Validasi Semua Model

```bash
python scripts/validate_models.py --all
```

### Validasi Model Spesifik

```bash
# Hanya NLP
python scripts/validate_models.py --nlp

# Hanya Vision
python scripts/validate_models.py --vision

# Model Typing Stress
python scripts/validate_models.py --typing

# Model Sleep Scoring
python scripts/validate_models.py --sleep

# Health Score
python scripts/validate_models.py --health
```

## 📊 Target Metrik (Threshold)

| Model                  | Metrik                   | Target |
| :--------------------- | :----------------------- | :----- |
| **NLP Stress/Emotion** | Accuracy                 | ≥ 85%  |
| **Food Vision**        | Classification Accuracy  | ≥ 85%  |
| **Food Vision**        | Calorie MAE (normalized) | ≤ 0.02 |
| **Typing Stress**      | AUC-ROC                  | ≥ 0.80 |
| **Sleep Scoring**      | MAE (normalized)         | ≤ 0.02 |
| **Health Score**       | MAE (normalized)         | ≤ 0.02 |

## 📝 Memahami Output

Script akan memberikan output berwarna untuk memudahkan review:

- `PASS` (Hijau): Model memenuhi target.
- `FAIL` (Merah): Model di bawah target, perlu training ulang atau tuning.
- `WARNING` (Kuning): File model atau data tidak ditemukan.

Jika ada model yang `FAIL`, script akan keluar dengan exit code `1`, yang berguna untuk integrasi CI/CD di masa depan.

## 🔧 Troubleshooting

- **Error: Custom components not found**: Pastikan file di `models/custom_*.py` tidak kosong dan berisi kelas yang dibutuhkan.
- **Error: Model not found**: Pastikan model telah di-download dari Colab dan diletakkan di folder `models/` sesuai nama yang diharapkan (e.g., `nlp_model`, `vision_model`).
- **Memory Error**: Jika validasi Vision gagal karena memory, pastikan tidak ada proses training lain yang berjalan di GPU/RAM.

---

**PIC:** Bagus & Putri
**Last Updated:** 8 Mei 2026
