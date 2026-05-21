---
title: Vitara AI Service
emoji: 🏃
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI backend for Vitara app (FastAPI + RAG)
---

# Vitara AI Service 🏃

Layanan backend AI untuk aplikasi **Vitara** — platform kesehatan holistik. Dibangun dengan **FastAPI**, **TensorFlow**, dan **Gemini RAG (ChromaDB)**.

## Endpoint Tersedia

| Endpoint | Metode | Deskripsi |
|---|---|---|
| `/` | GET | Health check — cek status service |
| `/predict/journal` | POST | Analisis NLP teks jurnal (emosi & stres) |
| `/predict/food` | POST | Deteksi makanan dari gambar (MobileNetV2 TFLite) |
| `/predict/sleep` | POST | Analisis kualitas tidur |
| `/predict/typing` | POST | Deteksi stres dari pola mengetik (LSTM) |
| `/health/score` | POST | Kalkulasi skor kesehatan holistik |
| `/companion/chat` | POST | Asisten AI (Gemini 3.1 Flash Lite + RAG via ChromaDB) — SSE Streaming |

## Environment Variables (Secrets)

Konfigurasi variabel berikut di **Settings → Variables and secrets** Space Anda:

| Variable | Wajib | Deskripsi |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Ya | API Key dari [Google AI Studio](https://aistudio.google.com/) untuk fitur LLM Companion |
| `TOPIC_MODEL_NAME` | Opsional | Model Gemma untuk ekstraksi topik (default: `gemma-4-31b`) |
| `APP_ENV` | Opsional | `production` atau `development` (default: `production`) |

> **Catatan:** ChromaDB berjalan secara lokal di dalam container. Data memori RAG akan ter-reset setiap kali Space di-restart (ephemeral storage). Untuk persistensi data, integrasikan dengan cloud vector database seperti Qdrant Cloud atau Pinecone.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **ML Runtime:** TensorFlow 2.16 (TFLite untuk inferensi vision & typing)
- **LLM:** Google Gemini 3.1 Flash Lite via `google-genai`
- **Vector DB (RAG):** ChromaDB (persistent in-container)
- **Deployment:** Docker (Hugging Face Spaces)
