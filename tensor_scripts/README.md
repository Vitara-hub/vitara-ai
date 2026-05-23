# Tensor Scripts

Folder ini berisi skrip utilitas bertenaga tinggi yang menggunakan TensorFlow/Keras untuk pengelolaan model (seperti konversi, pengujian, atau optimasi) tanpa mengotori environment runtime utama di `vitara-ai-service`. Dengan memisahkan skrip ini, backend API kita tetap ringan tanpa dependensi TensorFlow/Keras yang sangat besar.

## Skrip Konversi Keras ke ONNX (`convert_keras_to_onnx.py`)

Skrip ini dirancang khusus untuk mengonversi model Keras NLP IndoBERT (`.keras`) ke format ONNX (`.onnx`). Format ONNX ini nantinya akan dieksekusi menggunakan ONNX Runtime (`onnxruntime`), yang menawarkan kecepatan inferensi lebih tinggi serta pemakaian memori yang jauh lebih efisien pada backend API.

### Fitur Utama & Solusi Kompatibilitas

1. **Patch Deserialisasi Keras 3 untuk Model Keras 2 (Legacy)**
   Model Keras yang disimpan sering kali memiliki format konfigurasi legacy (Keras 2) yang tidak didukung langsung oleh Keras 3. Skrip ini menerapkan monkey-patching dinamis tingkat rendah pada parser Keras 3 (`deserialize_keras_object`) untuk:
   * Mengonversi format koneksi layer `inbound_nodes` yang usang (Keras 2) menjadi format `__keras_tensor__` yang kompatibel dengan Keras 3 secara dinamis.
   * Melakukan inferensi tipe data (dtype) dan bentuk dimensi (shape) layer secara rekursif selama proses pemuatan konfigurasi.
   * Memetakan nama kelas model fungsional legacy (`tf_keras.src.engine.functional.Functional`) ke kelas `Functional` bawaan Keras 3.
   * Menangani parameter list `axis` di `BatchNormalization` menjadi integer/tuple tunggal agar tidak terjadi error deserialisasi.
   * Menghapus konfigurasi kuantisasasi lama (`quantization_config`) dari model mentah yang tidak lagi didukung oleh layer Dense/LSTM Keras 3.

2. **Registrasi Custom Layers & Aktivasi**
   Mendaftarkan kelas dan fungsi kustom yang digunakan oleh model IndoBERT dan model klasifikasi teks:
   * **`AttentionLayer`** & **`WeightedFocalLoss`**: Diimpor secara dinamis dari kode sumber `vitara-ai-service` jika tersedia.
   * **`IndoBERTEncoder`**: Pembungkus bertipe Keras Layer untuk model BERT dari HuggingFace `transformers` (`TFBertModel` berbasis `indobenchmark/indobert-base-p2`).
   * **`gelu`**: Menggunakan aproksimasi tanh (`approximate=True`) untuk kompabilitas performa tinggi.
   * **`NotEqual`** & **`Any`**: Komponen logika tensor kustom untuk pra-pemrosesan data masukan.

3. **Deteksi Signature Input Dinamis**
   Skrip secara otomatis membaca seluruh input layer dari model Keras asli, lalu menyusun tanda tangan input (`input_signature`) dengan type-hinting dan penamaan tensor (`tf.TensorSpec`) yang presisi sebelum dikirimkan ke converter `tf2onnx`.

4. **Fase Verifikasi Otomatis dengan Data Tiruan**
   Setelah konversi selesai, skrip menjalankan fase validasi menggunakan `onnxruntime` untuk membandingkan output model Keras vs model ONNX:
   * Menghasilkan data dummy dinamis sesuai dengan shape input model.
   * **Batasan Masking BERT**: Nilai dummy untuk `token_type_ids` dan `attention_mask` dibatasi secara ketat pada rentang `[0, 1]` untuk mencegah kesalahan indeks di luar batas (`out-of-bounds index`) pada modul `TFBertModel`.
   * Memvalidasi hasil prediksi kedua model dengan batas toleransi selisih maksimum $1 \times 10^{-4}$ (0.0001).

---

### Cara Penggunaan

Ada dua cara untuk menjalankan skrip ini: menggunakan **`uv`** (sangat cepat dan direkomendasikan) atau menggunakan **`venv` + `pip`** biasa.

#### Opsi A: Menggunakan `uv` (Direkomendasikan)

Jika Anda sudah menginstal [`uv`](https://github.com/astral-sh/uv), Anda tidak perlu mengaktifkan virtual environment secara manual. Cukup jalankan skrip, dan `uv` akan otomatis menyiapkan dependensi yang sesuai:

```bash
# Masuk ke folder tensor_scripts
cd tensor_scripts

# Jalankan konversi dengan model & output default
uv run python convert_keras_to_onnx.py
```

Anda juga bisa menambahkan argumen kustom:
```bash
uv run python convert_keras_to_onnx.py \
  --input ../vitara-ai-service/models/nlp_model/vitara_nlp_indobert_20260522_0846.keras \
  --output ../vitara-ai-service/models/nlp_model/vitara_nlp_indobert.onnx \
  --opset 15
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

# Jalankan skrip konversi
python convert_keras_to_onnx.py
```

---

### Kustomisasi Command Line

Skrip mendukung argumen command line berikut untuk fleksibilitas:

| Argumen | Tipe | Default | Deskripsi |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | `../vitara-ai-service/models/nlp_model/vitara_nlp_indobert_20260522_0846.keras` | Lokasi file model Keras (`.keras` atau `.h5`) yang ingin dikonversi. |
| `--output` | `str` | `../vitara-ai-service/models/nlp_model/vitara_nlp_indobert.onnx` | Lokasi penyimpanan hasil konversi model ONNX. |
| `--opset` | `int` | `15` | Versi ONNX opset (disarankan 15 untuk mendukung operasi Transformer BERT). |

Untuk melihat bantuan command line:
```bash
python convert_keras_to_onnx.py --help
# atau jika memakai uv
uv run python convert_keras_to_onnx.py --help
```
