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

## 2. Model Typing Stress LSTM (ONNX)

Model pendeteksi tingkat stres pengguna berdasarkan pola pengetikan tombol keyboard (*keystroke dynamics*) menggunakan format ONNX (`.onnx`).

### Skrip
`inference_typing.py`

### Format Perintah

```bash
# Jika virtual environment sudah aktif:
python inference_typing.py '<json_payload>' [path_ke_model_onnx]

# Atau menggunakan uv run (otomatis mengaktifkan venv):
uv run python inference_typing.py '<json_payload>' [path_ke_model_onnx]
```

### Keterangan Parameter

- `<json_payload>`: String berformat JSON yang berisi data sesi mengetik pengguna (sesuai API contract):
  - `wpm` (float): Kecepatan mengetik dalam kata per menit (*Words Per Minute*).
  - `backspace_rate` (float): Rasio penekanan tombol backspace (0.0 – 1.0).
  - `inter_key_timings` (integer[]): List interval waktu antar ketukan tombol (milidetik).
- `[path_ke_model_onnx]`: _(Opsional)_ Lokasi file model `.onnx`. Secara default mencari ke `models/typing_model/typing_stress_lstm.onnx`.

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

## 3. Model NLP Stress/Emotion (ONNX)

Model pendeteksi emosi dan tingkat stres pengguna berdasarkan teks jurnal menggunakan format ONNX (`.onnx`) dan tokenizer lokal Hugging Face.

### Skrip
`inference_nlp.py`

### Format Perintah

```bash
# Jika virtual environment sudah aktif:
python inference_nlp.py '<teks_jurnal_atau_path_json>' [path_ke_model_onnx] [path_ke_folder_tokenizer]

# Atau menggunakan uv run (otomatis mengaktifkan venv):
uv run python inference_nlp.py '<teks_jurnal_atau_path_json>' [path_ke_model_onnx] [path_ke_folder_tokenizer]
```

### Keterangan Parameter

- `<teks_jurnal_atau_path_json>`: Teks jurnal dalam Bahasa Indonesia yang ingin dianalisis, ATAU lokasi/path file JSON yang berisi teks jurnal (contoh: `sample/nlp/happy_journal.json`).
- `[path_ke_model_onnx]`: _(Opsional)_ Lokasi file model `.onnx`. Secara default mencari ke `models/nlp_model/vitara_nlp_indobert.onnx`.
- `[path_ke_folder_tokenizer]`: _(Opsional)_ Lokasi folder tokenizer. Secara default mencari ke `models/nlp_model/tokenizer/`.

### Pengujian Menggunakan Data Sampel

Telah disediakan beberapa data sampel di folder `sample/nlp/` untuk memudahkan verifikasi emosi dan tingkat stres.

**1. Menggunakan Sampel Emosi Senang (Happy - Low Stress):**
```bash
# Menggunakan python standar (venv aktif):
python inference_nlp.py sample/nlp/happy_journal.json

# Menggunakan uv run:
uv run python inference_nlp.py sample/nlp/happy_journal.json
```

**2. Menggunakan Sampel Emosi Cemas (Anxious - High Stress):**
```bash
# Menggunakan uv run:
uv run python inference_nlp.py sample/nlp/anxious_journal.json
```

**3. Menggunakan Sampel Emosi Sedih (Sad):**
```bash
# Menggunakan uv run:
uv run python inference_nlp.py sample/nlp/sad_journal.json
```

**4. Menggunakan Sampel Emosi Marah (Angry):**
```bash
# Menggunakan uv run:
uv run python inference_nlp.py sample/nlp/angry_journal.json
```

**5. Menggunakan Sampel Emosi Netral (Neutral):**
```bash
# Menggunakan uv run:
uv run python inference_nlp.py sample/nlp/neutral_journal.json
```

**6. Menggunakan Input Kustom Manual (Inline Teks):**

Jika Anda ingin melakukan pengujian langsung dengan mengetikkan kalimat kustom di terminal tanpa membuat file baru:

```bash
# Menggunakan python standar (venv aktif):
python inference_nlp.py 'Hari ini saya merasa sangat cemas karena tugas menumpuk.'

# Menggunakan uv run:
uv run python inference_nlp.py 'Hari ini saya merasa sangat cemas karena tugas menumpuk.'
```

**Output (Stdout JSON):**
```json
{
  "emotion": "anxious",
  "stress_level": 0.82,
  "emotion_probabilities": {
    "angry": 0.05,
    "anxious": 0.82,
    "happy": 0.01,
    "neutral": 0.10,
    "sad": 0.02
  }
}
```

---

## 4. Model Sleep Quality (TFLite)

Model pendeteksi skor kualitas tidur pengguna (skala 0 - 100) menggunakan format TensorFlow Lite (`.tflite`).

### Skrip
`inference_sleep.py`

### Format Perintah

```bash
# Jika virtual environment sudah aktif:
python inference_sleep.py '<json_payload_atau_path_json>' [path_ke_model_tflite]

# Atau menggunakan uv run:
uv run python inference_sleep.py '<json_payload_atau_path_json>' [path_ke_model_tflite]
```

### Keterangan Parameter

- `<json_payload_atau_path_json>`: String berformat JSON yang berisi data log tidur pengguna, ATAU lokasi/path file JSON yang berisi data tidur (contoh: `sample/sleep/good_sleep.json`):
  - `duration_hours` (float): Durasi tidur dalam jam.
  - `interruptions` (integer): Frekuensi terbangun di malam hari.
  - `sleep_debt_hours` (float, opsional): Jam utang tidur yang diakumulasi (jika kosong, otomatis dianggap `0.0` oleh model).
- `[path_ke_model_tflite]`: _(Opsional)_ Lokasi file model `.tflite`. Secara default mencari ke `models/sleep_model/sleep_scoring_model.tflite`.

### Pengujian Menggunakan Data Sampel

Telah disediakan beberapa data sampel di folder `sample/sleep/` untuk memudahkan verifikasi.

**1. Menggunakan Sampel Tidur Nyenyak (Good Sleep):**
```bash
# Menggunakan python standar (venv aktif):
python inference_sleep.py "$(cat sample/sleep/good_sleep.json)"

# Menggunakan uv run:
uv run python inference_sleep.py "$(cat sample/sleep/good_sleep.json)"
```

**2. Menggunakan Sampel Tidur Buruk (Poor Sleep):**
```bash
# Menggunakan uv run:
uv run python inference_sleep.py "$(cat sample/sleep/poor_sleep.json)"
```

**3. Menggunakan Sampel Tidur Sedang dengan Utang Tidur (Average Sleep):**
```bash
# Menggunakan uv run:
uv run python inference_sleep.py "$(cat sample/sleep/average_sleep_with_debt.json)"
```

**4. Menggunakan Input Kustom Manual (Inline Teks):**

Jika Anda ingin melakukan pengujian langsung dengan mengetikkan JSON kustom di terminal tanpa membuat file baru:

```bash
# Menggunakan python standar (venv aktif):
python inference_sleep.py '{"duration_hours": 6.5, "interruptions": 2, "sleep_debt_hours": 1.0}'

# Menggunakan uv run:
uv run python inference_sleep.py '{"duration_hours": 6.5, "interruptions": 2, "sleep_debt_hours": 1.0}'
```

**Output (Stdout JSON):**
```json
{
  "quality_score": 75
}
```

