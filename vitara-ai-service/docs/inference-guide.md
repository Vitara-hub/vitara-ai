# 💻 Vitara AI — Standalone CLI Inference Guide

Dokumen ini menyediakan panduan lengkap untuk menjalankan skrip inferensi mandiri (*standalone CLI inference*) untuk setiap model AI di ekosistem Vitara AI secara langsung melalui terminal. 

Skrip inferensi mandiri ini sangat berguna untuk keperluan pengujian performa, verifikasi model, maupun integrasi *lightweight* tanpa perlu menjalankan server web FastAPI secara penuh.

---

## Persiapan Awal

Pastikan *virtual environment* Anda sudah aktif sebelum menjalankan perintah di bawah ini:

```bash
# Mengaktifkan venv (macOS/Linux)
source .venv/bin/activate

# Mengaktifkan venv (Windows)
.venv\Scripts\activate
```

Atau gunakan `uv run` untuk menjalankannya secara otomatis menggunakan *environment* yang terisolasi.

---

## 1. Model Food Vision (TFLite)

Model pendeteksi kalori dan jenis makanan berbasis citra digital menggunakan TensorFlow Lite (`.tflite`).

### Skrip
`inference_food.py`

### Format Perintah

```bash
# Jika virtual environment sudah aktif:
python inference_food.py <path_ke_gambar> <path_ke_model_tflite> [path_ke_file_classes_txt]

# Atau menggunakan uv run (otomatis mengaktifkan venv):
uv run python inference_food.py <path_ke_gambar> <path_ke_model_tflite> [path_ke_file_classes_txt]
```

### Keterangan Parameter

- `<path_ke_gambar>`: Lokasi file gambar makanan yang ingin dideklarasikan (contoh: `sample/food_vision/ayam_geprek.jpg`).
- `<path_ke_model_tflite>`: Lokasi model `.tflite` (contoh: `models/vision_model/vision_model.tflite`).
- `[path_ke_file_classes_txt]`: _(Opsional)_ Lokasi file teks berisi nama-nama label kelas makanan. Secara default mengacu pada `classes.txt`.

### Contoh Penggunaan

```bash
# Menggunakan python standar (venv aktif):
python inference_food.py sample/food_vision/ayam_geprek.jpg ./models/vision_model/vision_model.tflite ./models/vision_model/classes.txt

# Menggunakan uv run:
uv run python inference_food.py sample/food_vision/ayam_geprek.jpg ./models/vision_model/vision_model.tflite ./models/vision_model/classes.txt
```

**Output (Stdout JSON):**
```json
{
  "foods": ["nasi goreng"],
  "estimated_calories": 520
}
```

---

## 2. Model Typing Stress LSTM (H5)

Model pendeteksi tingkat stres pengguna berdasarkan pola pengetikan tombol keyboard (*keystroke dynamics*) menggunakan Keras H5 format (`.h5`).

### Skrip
`inference_typing.py`

### Format Perintah

```bash
# Jika virtual environment sudah aktif:
python inference_typing.py '<json_payload>' [path_ke_model_h5]

# Atau menggunakan uv run (otomatis mengaktifkan venv):
uv run python inference_typing.py '<json_payload>' [path_ke_model_h5]
```

### Keterangan Parameter

- `<json_payload>`: String berformat JSON yang berisi data sesi mengetik pengguna (sesuai API contract):
  - `wpm` (float): Kecepatan mengetik dalam kata per menit (*Words Per Minute*).
  - `backspace_rate` (float): Rasio penekanan tombol backspace (0.0 – 1.0).
  - `inter_key_timings` (integer[]): List interval waktu antar ketukan tombol (milidetik).
- `[path_ke_model_h5]`: _(Opsional)_ Lokasi file model `.h5`. Secara default mencari ke `models/typing_model/typing_stress_lstm.h5`.

### Pengujian Menggunakan Data Sampel

Telah disediakan data sampel di folder `sample/typing/` untuk memudahkan verifikasi.

**1. Menggunakan Data Sampel Rileks (Low Stress):**
```bash
# Menggunakan python standar (venv aktif):
python inference_typing.py "$(cat sample/typing/normal_typing.json)"

# Menggunakan uv run:
uv run python inference_typing.py "$(cat sample/typing/normal_typing.json)"
```

**2. Menggunakan Data Sampel Stres (High Stress):**
```bash
# Menggunakan python standar (venv aktif):
python inference_typing.py "$(cat sample/typing/stress_typing.json)"

# Menggunakan uv run:
uv run python inference_typing.py "$(cat sample/typing/stress_typing.json)"
```

**3. Menggunakan Data Sampel Stres Sedang (Moderate Stress):**
```bash
# Menggunakan python standar (venv aktif):
python inference_typing.py "$(cat sample/typing/moderate_typing.json)"

# Menggunakan uv run:
uv run python inference_typing.py "$(cat sample/typing/moderate_typing.json)"
```

**4. Menggunakan Input Kustom Manual (Opsional):**

Jika Anda ingin melakukan pengujian mandiri menggunakan data kustom langsung tanpa membuat berkas JSON baru di folder `sample/`, Anda dapat menuliskan JSON string secara inline:

```bash
# Menggunakan python standar (venv aktif):
python inference_typing.py '{"wpm": 58.3, "backspace_rate": 0.12, "inter_key_timings": [120, 98, 145, 87, 203, 110]}'

# Menggunakan uv run:
uv run python inference_typing.py '{"wpm": 58.3, "backspace_rate": 0.12, "inter_key_timings": [120, 98, 145, 87, 203, 110]}'
```

**Output (Stdout JSON):**
```json
{
  "stress_score": 0.74
}
```

---

## 3. Model NLP Stress/Emotion & Sleep (Dalam Pengembangan)

Skrip inferensi mandiri untuk model NLP (`inference_nlp.py`) dan kualitas tidur (`inference_sleep.py`) saat ini sedang dalam proses penyelarasan dengan arsitektur backend terbaru. Detil penggunaan akan ditambahkan setelah model selesai divalidasi.
