# 📦 Vitara AI — Dataset Contract

> **Authors:** Bagus + Putri  
> **Last Updated:** April 2026  
> **Status:** Draft — Untuk dikonfirmasi bersama tim Data Science

---

## Tujuan Dokumen

Dokumen ini mendefinisikan **format, skema, dan syarat dataset** yang dibutuhkan oleh tim AI dari tim Data Science. Harap dipastikan dataset yang dikirim sesuai dengan spesifikasi di bawah sebelum di-handover.

---

## 1. Dataset NLP — Journal Text

**PIC AI:** Putri  
**Lokasi target:** `data/nlp/raw/journals.csv`

### Format

File tunggal berformat **CSV** dengan encoding **UTF-8**.

### Skema Kolom

| Kolom           | Type     | Required | Contoh Nilai                            | Keterangan                                                                      |
| --------------- | -------- | -------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| `id`            | `string` | ✅       | `"jrn_001"`                             | ID unik per entri jurnal                                                        |
| `text`          | `string` | ✅       | `"Hari ini aku merasa sangat lelah..."` | Teks jurnal pengguna. **Tidak boleh kosong.**                                   |
| `emotion_label` | `string` | ✅       | `"anxious"`                             | Label emosi dominan. Nilai valid: `happy`, `sad`, `anxious`, `angry`, `neutral` |
| `stress_label`  | `float`  | ✅       | `0.8`                                   | Tingkat stres, skala **0.0 – 1.0**                                              |
| `language`      | `string` | ❌       | `"id"`                                  | Kode bahasa ISO 639-1. Default: `"id"` (Bahasa Indonesia)                       |

### Contoh Isi File

```csv
id,text,emotion_label,stress_label,language
jrn_001,"Hari ini aku merasa sangat lelah dan tertekan karena deadline menumpuk.",anxious,0.85,id
jrn_002,"Akhirnya liburan tiba! Aku sangat senang bisa istirahat panjang.",happy,0.10,id
jrn_003,"Aku tidak tahu harus bagaimana lagi, semuanya terasa berat.",sad,0.72,id
```

### Syarat Tambahan

- Minimal **5.000 entri** untuk pelatihan yang memadai.
- Split yang diharapkan: **80% train / 10% val / 10% test** (boleh dilakukan oleh tim AI).
- Distribusi label `emotion_label` harus **seimbang** atau disertai catatan distribusinya.
- Tidak boleh ada nilai `null` atau string kosong pada kolom `text` dan `emotion_label`.

---

## 2. Dataset Vision — Food Images

**PIC AI:** Bagus  
**Lokasi target:** `data/vision/raw/`

### Format

Struktur folder per kelas (label), masing-masing folder berisi file gambar.

```
data/vision/raw/
├── nasi_goreng/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── ayam_bakar/
│   ├── img_001.jpg
│   └── ...
└── ...
```

### Syarat Gambar

| Properti         | Syarat                                |
| ---------------- | ------------------------------------- |
| Format file      | JPEG atau PNG                         |
| Ukuran file      | Maks **2MB** per gambar               |
| Resolusi minimal | **224 × 224** piksel                  |
| Kondisi gambar   | Fokus pada makanan, pencahayaan cukup |

### Metadata Kalori

Sertakan file **`calorie_map.csv`** yang memetakan nama kelas ke estimasi kalori per 100g.

```csv
class_name,calories_per_100g
nasi_goreng,180
ayam_bakar,165
gado_gado,120
```

| Kolom               | Type      | Required | Keterangan                           |
| ------------------- | --------- | -------- | ------------------------------------ |
| `class_name`        | `string`  | ✅       | Harus sama persis dengan nama folder |
| `calories_per_100g` | `integer` | ✅       | Kalori per 100 gram dalam kkal       |

### Syarat Tambahan

- Minimal **500 gambar per kelas**, minimal **10 kelas** makanan.
- Tidak ada duplikat gambar antar kelas.
- Sertakan file `classes.txt` berisi daftar semua nama kelas (satu per baris).

---

## 3. Dataset Sleep — Sleep Logs

**PIC AI:** Putri  
**Lokasi target:** `data/sleep/raw/sleep_logs.csv`

### Format

File tunggal berformat **CSV** dengan encoding **UTF-8**.

### Skema Kolom

| Kolom              | Type      | Required | Contoh Nilai   | Keterangan                               |
| ------------------ | --------- | -------- | -------------- | ---------------------------------------- |
| `id`               | `string`  | ✅       | `"slp_001"`    | ID unik per entri                        |
| `user_id`          | `string`  | ✅       | `"usr_abc"`    | ID anonim pengguna                       |
| `date`             | `string`  | ✅       | `"2025-03-01"` | Tanggal tidur, format `YYYY-MM-DD`       |
| `bedtime`          | `string`  | ✅       | `"23:30"`      | Waktu mulai tidur, format `HH:MM`        |
| `wake_time`        | `string`  | ✅       | `"06:00"`      | Waktu bangun, format `HH:MM`             |
| `duration_hours`   | `float`   | ✅       | `6.5`          | Total durasi tidur dalam jam             |
| `interruptions`    | `integer` | ✅       | `2`            | Jumlah kali terbangun                    |
| `sleep_debt_hours` | `float`   | ❌       | `1.5`          | Akumulasi utang tidur dalam jam          |
| `quality_score`    | `float`   | ✅       | `0.72`         | Label skor kualitas tidur, **0.0 – 1.0** |

### Contoh Isi File

```csv
id,user_id,date,bedtime,wake_time,duration_hours,interruptions,sleep_debt_hours,quality_score
slp_001,usr_abc,2025-03-01,23:30,06:00,6.5,2,1.5,0.72
slp_002,usr_def,2025-03-01,01:00,05:30,4.5,5,3.0,0.38
```

### Syarat Tambahan

- Minimal **3.000 entri** dari **minimal 100 pengguna unik**.
- Kolom `quality_score` adalah **ground truth label** — pastikan metodologi pelabelannya didokumentasikan.

---

## 4. Dataset Typing — Keystroke Dynamics

**PIC AI:** Bagus  
**Lokasi target:** `data/typing/raw/keystrokes.csv`

### Format

File tunggal berformat **CSV** dengan encoding **UTF-8**.

### Skema Kolom

| Kolom               | Type     | Required | Contoh Nilai            | Keterangan                                   |
| ------------------- | -------- | -------- | ----------------------- | -------------------------------------------- |
| `id`                | `string` | ✅       | `"typ_001"`             | ID unik per sesi                             |
| `user_id`           | `string` | ✅       | `"usr_abc"`             | ID anonim pengguna                           |
| `wpm`               | `float`  | ✅       | `58.3`                  | Kecepatan mengetik (kata per menit)          |
| `typing_variance`   | `float`  | ✅       | `15.4`                  | Variansi (Standard Deviation) dalam ms       |
| `backspace_rate`    | `float`  | ✅       | `0.12`                  | Rasio backspace (0.0 – 1.0)                  |
| `inter_key_timings` | `string` | ✅       | `"120,98,145,87"`       | Interval antar tombol dalam ms, dipisah koma |
| `stress_label`      | `float`  | ✅       | `0.74`                  | Label tingkat stres, **0.0 – 1.0**           |

### Contoh Isi File

```csv
id,user_id,wpm,typing_variance,backspace_rate,inter_key_timings,stress_label
typ_001,usr_abc,58.3,15.4,0.12,"120,98,145,87,203,110",0.74
typ_002,usr_def,42.1,32.8,0.25,"230,198,310,145,280",0.88
```

### Syarat Tambahan

- Minimal **2.000 sesi** dari **minimal 50 pengguna unik**.
- Kolom `inter_key_timings` disimpan sebagai string dengan nilai dipisah koma.
- Sertakan catatan bagaimana `stress_label` ditentukan (self-report, sensor, dll.).

---

## 5. Dataset Health Score — Ground Truth Labels

**PIC AI:** Putri  
**Lokasi target:** `data/health_score/raw/health_labels.csv`

### Format

File tunggal berformat **CSV** dengan encoding **UTF-8**.

### Skema Kolom

| Kolom          | Type     | Required | Contoh Nilai | Keterangan                                                                  |
| -------------- | -------- | -------- | ------------ | --------------------------------------------------------------------------- |
| `user_id`      | `string` | ✅       | `"usr_abc"`  | ID unik pengguna                                                            |
| `date`         | `string` | ✅       | `2025-03-01` | Tanggal penilaian                                                           |
| `health_score` | `float`  | ✅       | `78.5`       | **Ground Truth Health Score** (Label), skala **0.0 - 100.0**                |
| `mood_label`   | `float`  | ❌       | `70.0`       | Skor dimensi emosi (opsional)                                               |
| `stress_label` | `float`  | ❌       | `65.0`       | Skor dimensi stres (opsional)                                               |
| `sleep_label`  | `float`  | ❌       | `72.0`       | Skor dimensi tidur (opsional)                                               |
| `source`       | `string` | ✅       | `"PHQ-9"`    | Sumber penilaian (e.g. kuesioner medis, kuesioner stress, penilaian dokter) |

### Contoh Isi File

```csv
user_id,date,health_score,mood_label,stress_label,sleep_label,source
usr_abc,2025-03-01,78.5,70,65,72,Kuesioner_A
usr_def,2025-03-01,45.0,40,30,50,Penilaian_Dokter
```

### Syarat Tambahan

- Minimal **1.000 data point** yang berpasangan dengan data (NLP, Food, Sleep, Typing) pada rentang waktu yang sama.
- Label `health_score` harus berasal dari instrumen penilaian yang valid dan terstandarisasi.

---

> _Hubungi Tim AI jika ada pertanyaan mengenai spesifikasi di atas._
