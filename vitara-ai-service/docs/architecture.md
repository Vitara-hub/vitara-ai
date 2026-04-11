# 🏗️ Vitara AI — Architecture Document

> **Authors:** Bagus
> **Last Updated:** April 2026  
> **Status:** Draft

---

## Gambaran Umum Sistem

Vitara AI Service adalah backend berbasis **FastAPI** yang meng-expose endpoint inferensi model machine learning. Setiap model dikembangkan secara independen lalu diintegrasikan ke dalam satu layanan terpadu.

```
[Mobile App / Frontend Next.js]
          │
          ▼
  [Vitara AI Service — FastAPI]
  ┌──────────────────────────────────────────┐
  │  POST /predict/journal   → NLP Model     │
  │  POST /predict/food      → Vision Model  │
  │  POST /predict/sleep     → Sleep Model   │
  │  POST /predict/typing    → Typing Model  │
  │  POST /health/score      → Health Score  │
  │  POST /companion/chat    → LLM Companion │
  └──────────────────────────────────────────┘
          │
          ▼
  [TensorFlow / TFLite Models]  +  [LLM API (Gemini/Claude)]
```

---

## Model Architectures

### 1. NLP Stress/Emotion Model

**Tugas:** Menganalisis teks jurnal pengguna untuk mendeteksi emosi dominan dan tingkat stres.

```
Input (raw text)
  └─► TextVectorization
        └─► Embedding
              └─► BiLSTM
                    └─► AttentionLayer       ← Custom Layer
                          ├─► emotion_head   → softmax  (multi-class emotion)
                          └─► stress_head    → sigmoid  (regression 0–1)
```

**Custom Components:**
| Komponen | Tipe |
|----------|------|
| `AttentionLayer` | Custom Layer |
| `WeightedFocalLoss` | Custom Loss |
| `VitaraTrainingLogger` | Custom Callback |
| `tf.GradientTape` loop | Custom Training Loop |

**Input:** Raw text string  
**Output:**

```json
{
  "emotion": "anxious",
  "stress_level": 0.82,
  "topics": ["deadline", "kerja"]
}
```

---

### 2. Food Vision Model

**Tugas:** Mengenali jenis makanan dari gambar dan memperkirakan estimasi kalori.

```
Input (image 224×224)
  └─► MobileNetV2 (pretrained, freeze awal)
        └─► GlobalAveragePooling2D
              └─► Dense
                    ├─► classification_head  → softmax  (nama makanan)
                    └─► calorie_head         → linear   (estimasi kalori)
```

> Fine-tuning: unfreeze layer atas MobileNetV2 setelah epoch awal konvergen.

**Input:** Gambar (base64 / multipart)  
**Output:**

```json
{
  "foods": ["nasi goreng", "telur"],
  "estimated_calories": 520
}
```

---

### 3. Typing Stress LSTM

**Tugas:** Menganalisis pola pengetikan (keystroke dynamics) untuk mendeteksi tingkat stres.

```
Input (keystroke sequence: WPM, backspace_rate, inter_key_timing)
  └─► LSTM / BiLSTM
        └─► Dense
              └─► stress_score → sigmoid (0–1)
```

**Input:** Array sequence keystroke metrics  
**Output:**

```json
{
  "stress_score": 0.74
}
```

---

### 4. Sleep Scoring Model

**Tugas:** Mengevaluasi kualitas pola tidur pengguna.

```
Input (sleep features: duration, interruptions, schedule regularity, etc.)
  └─► Dense(64)
        └─► Dense(32)
              └─► quality_score → linear (0–100)
```

**Input:** Sleep log features  
**Output:**

```json
{
  "quality_score": 72
}
```

---

### 5. Multimodal Health Score Model

**Tugas:** Menggabungkan output dari semua model menjadi skor kesehatan holistik menggunakan strategi **late fusion**.

```
nlp_embed     ─┐
vision_embed  ─┤
typing_embed  ─┼─► Concatenate → Dense(128) → Dense(64) → health_score (linear 0–100)
sleep_embed   ─┘
```

**Input:** Embedding/output dari NLP, Vision, Typing, dan Sleep model  
**Output:**

```json
{
  "health_score": 78,
  "breakdown": {
    "mood": 70,
    "nutrition": 85,
    "stress": 65,
    "sleep": 72
  }
}
```

---

### 6. LLM Companion

**Tugas:** Memberikan respons percakapan yang personal dan kontekstual berdasarkan data kesehatan pengguna menggunakan pipeline **RAG (Retrieval-Augmented Generation)**.

```
User Message
  └─► context_builder.py    (query ChromaDB → ambil memori relevan user)
        └─► prompts.py       (bangun system prompt + context window)
              └─► LLM API    (Gemini / Claude)
                    └─► response + recommendations
```

**Memory Store (ChromaDB):**

- Koleksi: `user_memories`
- Metadata per dokumen: `{ user_id, timestamp, type }`

**Input:** `{ user_id, message }`  
**Output:**

```json
{
  "response": "Sepertinya kamu cukup lelah hari ini, yuk istirahat sebentar...",
  "recommendations": ["Tidur lebih awal", "Kurangi kafein"]
}
```

---

## Target Metrik

| Model              | Metrik                   | Target    |
| ------------------ | ------------------------ | --------- |
| NLP Stress/Emotion | Accuracy                 | ≥ 85%     |
| Food Vision        | Classification Accuracy  | ≥ 85%     |
| Food Vision        | Calorie MAE (normalized) | ≤ 0.02    |
| Typing Stress LSTM | AUC-ROC                  | ≥ 0.80    |
| Sleep Scoring      | MAE (normalized)         | ≤ 0.02    |
| Health Score       | MAE (normalized)         | ≤ 0.02    |
| Semua endpoint API | Latency                  | ≤ 2 detik |

---

> _Dokumen ini bersifat living document dan akan terus diperbarui seiring perkembangan project._
