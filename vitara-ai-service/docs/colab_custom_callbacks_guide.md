# Panduan Menggunakan Custom Components di Google Colab

Dokumen ini menjelaskan cara menggunakan komponen kustom yang dikembangkan secara lokal (seperti `VitaraTrainingLogger`, custom layer, atau custom loss) ke dalam environment Google Colab untuk keperluan training model.

## Langkah 1: Push Perubahan ke GitHub

Pastikan semua file custom component (seperti `models/custom_callbacks.py`) sudah di-commit dan di-push ke repository GitHub agar bisa diunduh oleh Google Colab.

```bash
git add vitara-ai-service/models/custom_callbacks.py
git commit -m "feat: add VitaraTrainingLogger custom callback"
git push origin main
```

## Langkah 2: Clone Repository di Google Colab

Di sel pertama notebook Colab (misalnya `notebooks/vision_training.ipynb`), jalankan perintah berikut untuk mengambil kode terbaru dari repository.

```python
# ==========================================
# 1. SETUP & CLONE DARI GITHUB PRIVATE REPO
# ==========================================
import sys
import os
from getpass import getpass

print("Masukkan GitHub Personal Access Token (PAT) Anda:")
token = getpass()
repo_url = f"https://{token}@github.com/Vitara-hub/vitara-ai.git"

!rm -rf vitara-ai
!git clone {repo_url}
del token

# ==========================================
# 2. SETUP PATH UNTUK CUSTOM MODULE
# ==========================================
service_path = '/content/vitara-ai/vitara-ai-service'
if service_path not in sys.path:
    sys.path.append(service_path)
    print("✅ Berhasil menambahkan path vitara-ai-service ke sys.path")
```

## Langkah 4: Import dan Gunakan Callback

Sekarang bisa mengimpor dan menggunakan `VitaraTrainingLogger` seperti modul Python biasa.

**Catatan:** Saat memanggil `model.fit()`, ubah argumen `verbose=0` agar output log default dari TensorFlow (yang berupa progress bar) disembunyikan, sehingga log terlihat lebih bersih dan rapi.

```python
import tensorflow as tf
import sys

# Tambahkan path folder 'models' langsung ke sys.path
sys.path.insert(0, '/content/vitara-ai/vitara-ai-service/models')
from custom_callbacks import VitaraTrainingLogger

# Inisialisasi logger (misal: print log setiap 1 epoch)
v_logger = VitaraTrainingLogger(log_frequency=1)

# Training model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    verbose=0, # <-- MATIKAN VERBOSE DEFAULT TENSORFLOW
    callbacks=[v_logger] # <-- TAMBAHKAN LOGGER KE DALAM LIST CALLBACKS
)
```

## Contoh Output Log

Saat proses training berjalan, akan melihat log yang diformat dengan cantik seperti ini:

```text
================================================================================
🚀 [Vitara AI] Training Started at 2026-05-06 14:00:00
================================================================================
Epoch 001 | 45.2s | loss: 0.8541 | accuracy: 0.7210 | val_loss: 0.5123 | val_accuracy: 0.8123
Epoch 002 | 43.1s | loss: 0.4215 | accuracy: 0.8544 | val_loss: 0.3842 | val_accuracy: 0.8655
...
================================================================================
✅ [Vitara AI] Training Completed in 00:15:32.4
================================================================================
```

## FAQ / Troubleshooting

1. **ModuleNotFoundError: No module named 'models'**
   - Pastikan sel yang menambahkan `sys.path` (Langkah 3) sudah dijalankan sebelum melakukan import.
2. **File tidak update meski sudah diubah di lokal?**
   - Pastikan perubahan lokal sudah di-push ke GitHub.
   - Di Colab, harus menjalankan ulang sel `!rm -rf vitara-ai` dan `!git clone ...` agar mengambil kode versi terbaru.
