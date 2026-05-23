# Tensor Scripts

Folder ini berisi skrip utilitas bertenaga tinggi yang menggunakan TensorFlow/Keras untuk pengelolaan model (seperti konversi, pengujian, atau optimasi) tanpa mengotori environment runtime utama di `vitara-ai-service`. Dengan memisahkan skrip ini, backend API kita tetap ringan tanpa dependensi TensorFlow/Keras yang sangat besar.

## Daftar Skrip Konversi

### 1. Konversi Model NLP IndoBERT (`convert_keras_to_onnx.py`)

Skrip ini digunakan untuk mengonversi model Keras NLP IndoBERT (`.keras`) ke format ONNX (`.onnx`) agar dapat dijalankan menggunakan ONNX Runtime yang lebih ringan dan efisien untuk inferensi di backend API.

#### Fitur Utama & Solusi Kompatibilitas
* **Patch Deserialisasi Keras 3**: Menerapkan monkey-patching dinamis pada parser Keras 3 (`deserialize_keras_object`) untuk memuat model Keras 2 (legacy) secara aman (resolusi `inbound_nodes`, `axis` di `BatchNormalization`, class name `Functional`, dan pembersihan `quantization_config`).
* **Registrasi Custom Layers**: Mendaftarkan custom layer (`AttentionLayer`, `WeightedFocalLoss`, `IndoBERTEncoder`, `gelu`, `NotEqual`, `Any`).
* **Deteksi Signature Input Dinamis**: Mendeteksi input signature model Keras secara dinamis.
* **Fase Verifikasi Otomatis**: Menjalankan dummy inference menggunakan data acak pada model Keras dan ONNX (`onnxruntime`), lalu membandingkan selisih outputnya (toleransi $< 10^{-4}$). Segment IDs dan attention masks dibatasi pada `[0, 1]` untuk menghindari error indeks pada BERT.

---

### 2. Konversi Model Typing Stress (`convert_typing_to_onnx.py`)

Skrip ini digunakan untuk mengonversi model klasifikasi stres mengetik berbasis LSTM (`.h5`) ke format ONNX (`.onnx`).

#### Fitur Utama & Solusi Kompatibilitas
* **Patch Deserialisasi Keras 3**: Sama seperti skrip NLP, skrip ini menggunakan parser dengan monkey-patching dinamis untuk memuat model Keras legacy yang menggunakan custom layers tanpa mengalami kegagalan/error kompatibilitas.
* **Registrasi Custom Layers**: Khusus mendaftarkan custom layer yang dipakai oleh model typing: `NotEqual` dan `Any` serta kelas `Dense`, `LSTM`, dan `Masking` yang dipatch untuk membuang `quantization_config` lama.
* **Input Signature Ganda**: Mendukung deteksi dinamis atau menggunakan fallback default input model mengetik:
  * `seq_input` dengan dimensi `(None, 50, 1)` bertipe `float32`.
  * `static_input` dengan dimensi `(None, 3)` bertipe `float32`.
* **Fase Verifikasi Otomatis**: Menguji keakuratan keluaran model ONNX vs Keras menggunakan data masukan acak multi-input (`seq_input` & `static_input`).

---

### Cara Penggunaan

Ada dua cara untuk menjalankan skrip-skrip ini: menggunakan **`uv`** (sangat cepat dan direkomendasikan) atau menggunakan **`venv` + `pip`** biasa.

#### Opsi A: Menggunakan `uv` (Direkomendasikan)

Jika Anda sudah menginstal [`uv`](https://github.com/astral-sh/uv), Anda tidak perlu mengaktifkan virtual environment secara manual. Cukup jalankan skrip, dan `uv` akan otomatis menyiapkan dependensi yang sesuai:

##### A. Konversi Model NLP IndoBERT
```bash
# Masuk ke folder tensor_scripts
cd tensor_scripts

# Jalankan konversi dengan model & output default
uv run python convert_keras_to_onnx.py
```

##### B. Konversi Model Typing Stress
```bash
# Masuk ke folder tensor_scripts
cd tensor_scripts

# Jalankan konversi dengan model & output default
uv run python convert_typing_to_onnx.py
```

---

#### Opsi B: Menggunakan Standard `venv` & `pip`

Jika menggunakan Python Virtual Environment bawaan:

```bash
# Masuk ke folder tensor_scripts
cd tensor_scripts

# Buat virtual environment
python3 -m venv .venv

# Aktifkan virtual environment
# Pada macOS/Linux:
source .venv/bin/activate
# Pada Windows (Command Prompt):
# .venv\Scripts\activate.bat

# Update pip dan instal seluruh dependensi
pip install --upgrade pip
pip install -r requirements.txt

# Jalankan skrip konversi pilihan Anda
python convert_keras_to_onnx.py
# ATAU
python convert_typing_to_onnx.py
```

---

### Parameter Command Line

Kedua skrip mendukung parameter masukan kustom melalui CLI:

#### Model NLP IndoBERT CLI
```bash
uv run python convert_keras_to_onnx.py \
  --input ../vitara-ai-service/models/nlp_model/vitara_nlp_indobert_20260522_0846.keras \
  --output ../vitara-ai-service/models/nlp_model/vitara_nlp_indobert.onnx \
  --opset 15
```

#### Model Typing Stress CLI
```bash
uv run python convert_typing_to_onnx.py \
  --input ../vitara-ai-service/models/typing_model/typing_stress_lstm.h5 \
  --output ../vitara-ai-service/models/typing_model/typing_stress_lstm.onnx \
  --opset 13
```

Untuk melihat bantuan command line dari masing-masing skrip:
```bash
python convert_keras_to_onnx.py --help
python convert_typing_to_onnx.py --help
```
