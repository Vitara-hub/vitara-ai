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

#### A. Rencana Transisi (IndoBERT Transformer Model — Baru/Direncanakan)

Untuk meningkatkan akurasi analisis emosi dan tingkat stres dengan pemahaman semantik Bahasa Indonesia yang lebih baik, sistem direncanakan beralih ke arsitektur berbasis **IndoBERT** (`indobenchmark/indobert-base-p2`).

```
Input (raw text)
  └─► IndoBERT Tokenizer (BPE)
        └─► IndoBERT Encoder (12 Transformer Blocks)
              └─► Pooling ([CLS] Token Representation / 768-d)
                    ├─► Emotion Head (Dense + Dropout) → softmax (multi-class emotion)
                    └─► Stress Head (Dense + Dropout)  → sigmoid (regression 0–1)
```

**Spesifikasi IndoBERT:**
| Komponen | Spesifikasi / Tipe | Keterangan |
|----------|-------------------|------------|
| **Base Model** | `indobenchmark/indobert-base-p2` | Pre-trained BERT model untuk Bahasa Indonesia (124M params) |
| **Tokenizer** | IndoBERT Tokenizer (BPE) | Tokenisasi teks disesuaikan kosa kata Bahasa Indonesia |
| **Emotion Head** | Dense (768) ➔ Dropout (0.1) ➔ Dense (5) ➔ Softmax | Klasifikasi 5 kelas emosi (`happy`, `sad`, `anxious`, `angry`, `neutral`) |
| **Stress Head** | Dense (768) ➔ Dropout (0.1) ➔ Dense (1) ➔ Sigmoid | Regresi nilai tingkat stres (skala 0.0 - 1.0) |
| **Loss Function** | Multi-Task Loss | Kombinasi `WeightedFocalLoss` (untuk emosi) + `MeanSquaredError` / `HuberLoss` (untuk stres) |
| **Optimizer** | AdamW | Dengan warm-up steps dan linear learning rate decay |

#### B. Model Awal / Baseline (BiLSTM + Attention Layer — Aktif/Legacy)

Pendekatan baseline saat ini menggunakan arsitektur recurrent neural network (RNN) berbasis Keras/TensorFlow yang ringan untuk inferensi awal:

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
**Output (API Response):**

```json
{
  "emotion": "anxious",
  "stress_level": 0.82,
  "topics": ["deadline", "kerja"]
}
```

> 📌 **Catatan Output:** 
> *   Nilai `emotion` dan `stress_level` diprediksi secara langsung oleh model NLP (BiLSTM/IndoBERT).
> *   Nilai `topics` diekstraksi di tingkat backend (*post-processing* menggunakan rule-based/keyword matching atau LLM) untuk mendukung konteks RAG pada LLM Companion. Oleh karena itu, data ini tidak memerlukan pelabelan pada dataset training model.


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

**Tugas:** Menggabungkan output dari semua model menjadi skor kesehatan holistik.

#### A. Rencana Awal (Trained Model Approach - Legacy/Planned)

Strategi awal menggunakan strategi **late fusion** berbasis neural network:

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

#### B. Pendekatan Aktif (Rule-Based Approach - Aktif)

Untuk menyederhanakan arsitektur dan meningkatkan transparansi perhitungan skor, sistem saat ini menggunakan pendekatan **Rule-Based (Berbasis Formula/Aturan)**.

Sistem menghitung sub-skor untuk masing-masing dimensi kesehatan terlebih dahulu sebelum menggabungkannya ke dalam nilai *Overall Health Score* berbasis rata-rata tertimbang (_weighted average_).

##### 1. Logika Perhitungan Sub-Skor (Dimensi):
*   **Mood Score (0 - 100)**: Dikonversi secara deterministik dari label emosi hasil analisis jurnal NLP:
    *   `happy` / `joy` / `excited` / `love` / `cheerful` = 90
    *   `neutral` / `calm` / `relaxed` = 70
    *   `sad` / `anxious` / `fear` / `stressed` / `lonely` / `worried` = 40
    *   `angry` / `frustrated` / `annoyed` / `irritated` = 30
    *   *Default/Lainnya* = 60
*   **Stress Score (0 - 100)**: Menggabungkan tingkat stres dari analisis jurnal (NLP) dan pola pengetikan (Keystroke Dynamics). Nilai dikonversi agar semakin tinggi nilai skor, semakin **tidak stres** (100 = bebas stres):
    *   `Stress_Score = 100 - ( (nlp.stress_level + typing.stress_score) / 2 * 100 )`
*   **Sleep Score (0 - 100)**: Diambil langsung dari nilai kualitas tidur:
    *   `Sleep_Score = sleep_result.quality_score`
*   **Nutrition Score (0 - 100)**: Mengevaluasi jumlah kalori estimasi per porsi makanan:
    *   *Ideal* (400 - 700 kkal) = 90
    *   *Kurang* (200 - 399 kkal) = 70
    *   *Sangat Kurang* (< 200 kkal) = 50
    *   *Berlebih* (701 - 1000 kkal) = 60
    *   *Sangat Berlebih* (> 1000 kkal) = 40

##### 2. Logika Perhitungan Overall Health Score:
Menggunakan rata-rata tertimbang (_weighted average_) dari keempat komponen utama di atas dengan distribusi bobot sebagai berikut:
*   **Sleep**: 30%
*   **Stress**: 30%
*   **Mood**: 20%
*   **Nutrition**: 20%

`Health_Score = (Sleep_Score * 0.3) + (Stress_Score * 0.3) + (Mood_Score * 0.2) + (Nutrition_Score * 0.2)`

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
**Output (Streaming SSE):**

*Event: `delta` (dikirim berkali-kali selama generasi kata/token)*
```json
{
  "token": "Sepertinya"
}
```

*Event: `final` (dikirim satu kali di akhir stream)*
```json
{
  "full_response": "Sepertinya kamu cukup lelah hari ini. Berdasarkan data tidurmu, kamu hanya tidur 5.5 jam semalam. Yuk coba istirahat lebih cepat malam ini!",
  "recommendations": [
    "Tidur lebih awal, target 7–8 jam",
    "Kurangi kafein setelah jam 3 sore",
    "Coba teknik pernapasan 4-7-8 sebelum tidur"
  ]
}
```

---

## Target Metrik

| Model              | Metrik                   | Target    |
| ------------------ | ------------------------ | --------- |
| NLP Stress/Emotion | Emotion Accuracy         | ≥ 85%     |
| NLP Stress/Emotion | Stress MAE (normalized)  | ≤ 0.15    |
| Food Vision        | Classification Accuracy  | ≥ 85%     |
| Food Vision        | Calorie MAE (normalized) | ≤ 0.02    |
| Typing Stress LSTM | Classification Accuracy  | ≥ 85%     |
| Typing Stress LSTM | Stress MAE (normalized)  | ≤ 0.02    |
| Sleep Scoring      | MAE (normalized)         | ≤ 0.15    |
| Semua endpoint API | Latency                  | ≤ 2 detik |

---

> _Dokumen ini bersifat living document dan akan terus diperbarui seiring perkembangan project._
