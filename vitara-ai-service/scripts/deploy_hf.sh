#!/bin/bash

# Konfigurasi path asal dan tujuan
SRC_DIR="/Users/bagus/Library/CloudStorage/OneDrive-UniversitasTerbuka/Stupen/Capstone Project/vitara-ai/vitara-ai-service/"
DEST_DIR="/Users/bagus/Library/CloudStorage/OneDrive-UniversitasTerbuka/Stupen/Capstone Project/vitara-ai-service/"

echo "🔄 1. Sinkronisasi file ke folder Hugging Face..."
rsync -av \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='data' \
  --exclude='logs' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.env' \
  --exclude='*.ipynb' \
  --exclude='*.keras' \
  --exclude='*.h5' \
  "$SRC_DIR" \
  "$DEST_DIR"

# Masuk ke direktori Hugging Face
cd "$DEST_DIR" || exit

# Rename README_HF.md jika ada
if [ -f README_HF.md ]; then
  echo "📄 2. Mengatur README.md..."
  rm -f README.md
  mv README_HF.md README.md
fi

echo "📤 3. Mengunggah perubahan ke Hugging Face..."

# Pastikan file besar selalu di-track oleh Git LFS
git lfs track "*.onnx" > /dev/null 2>&1
git lfs track "*.keras" > /dev/null 2>&1
git lfs track "*.h5" > /dev/null 2>&1
git lfs track "*.tflite" > /dev/null 2>&1
git add .gitattributes

# Re-register file onnx agar masuk LFS (bukan regular git)
git rm --cached models/nlp_model/vitara_nlp_indobert.onnx > /dev/null 2>&1 || true
git rm --cached models/typing_model/typing_stress_lstm.onnx > /dev/null 2>&1 || true

git add .

# Menggunakan pesan commit custom jika diberikan, jika tidak gunakan default
COMMIT_MSG=${1:-"Update service code and models"}
git commit -m "$COMMIT_MSG"

git push origin main --force

echo "✅ Selesai! Hugging Face Spaces akan mulai men-build ulang perubahan Anda secara otomatis."
