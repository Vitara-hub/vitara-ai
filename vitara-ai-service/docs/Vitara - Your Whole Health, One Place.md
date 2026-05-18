# Vitara - Your Whole Health, One Place

Platform kesehatan yang menggabungkan deteksi stres, kualitas tidur, gizi, dan mental health jadi satu ekosistem personal. Semua modul saling terhubung dan membentuk satu "health profile" yang berkembang tiap hari.

<aside>
🧠

**Mood & Mental Check**

Pengguna menulis jurnal harian bebas, lewat teks atau voice-to-text. Sistem lalu menganalisis pola bahasa, seperti keraguan, urgensi, dan pengulangan kata, untuk memetakan mood mingguan. Jika muncul pola tidak biasa, misalnya mood buruk beberapa hari berturut-turut, sistem mengirim pesan alert. Data ini juga dikaitkan dengan faktor lain, seperti kualitas tidur dan pengaruhnya terhadap mood hari berikutnya.

</aside>

<aside>
🌙

**Sleep Intelligence**

Pengguna mengisi jam tidur, jam bangun, dan kualitas tidur yang dirasakan. Sistem lalu menghubungkan data tersebut dengan mood, pola makan, dan tingkat stres untuk melihat hubungan yang sering muncul. Dari pola itu, sistem menyusun saran tidur yang sesuai dengan kebiasaan pengguna, misalnya: *“Kamu cenderung tidur kurang baik jika makan berat setelah jam 8 malam saat sedang stres.”* Saran ini diperbarui secara berkala berdasarkan data mingguan. 

</aside>

<aside>
🍽️

**Nutrition Lens**

Pengguna foto makanan, lalu sistem mengenali bahan dan estimasi porsinya untuk menghitung kandungan gizi. Data makanan ini kemudian dikaitkan dengan mood dan energi pada hari yang sama untuk melihat polanya, misalnya: *“Kamu cenderung merasa lemas di sore hari setelah makan tinggi karbohidrat saat siang.”* Hasilnya juga otomatis masuk ke penilaian kesehatan harian.

</aside>

<aside>
😮‍💨

**Stress Radar**

Pola mengetik seperti kecepatan, jeda, dan tingkat kesalahan dianalisis langsung di browser tanpa mengirim data ke luar perangkat. Jika tingkat stres terdeteksi tinggi, sistem memberi saran untuk berhenti sejenak dan mengatur napas. Data stres ini kemudian dihubungkan dengan modul lain, sehingga hari dengan stres tinggi ikut menjadi pertimbangan dalam jurnal mood dan rekomendasi tidur malam hari.

</aside>

---

### 👤 User Journey

```mermaid
graph TD
    %% Styling Classes
    classDef grey fill:#f2f1eb,stroke:#999,stroke-width:1px,color:#333;
    classDef blue fill:#ececff,stroke:#8a8df0,stroke-width:1px,color:#333;
    classDef green fill:#e2f5eb,stroke:#5ba889,stroke-width:1px,color:#333;
    classDef orange fill:#faebe6,stroke:#d98d75,stroke-width:1px,color:#333;
    classDef yellow fill:#fcefd9,stroke:#d9a86c,stroke-width:1px,color:#333;

    %% Nodes
    Start([User buka Vitara]):::grey
    
    Daily[<b>Daily check-in</b><br>Tulis jurnal / foto makan / log tidur]:::blue
    
    Mood[<b>Mood journal</b><br>Teks bebas / suara]:::blue
    Sleep[<b>Sleep log</b><br>Jam tidur + kualitas]:::green
    Food[<b>Foto makanan</b><br>Kamera langsung]:::orange
    Stress[<b>Stress radar</b><br>Keystroke pasif]:::yellow
    
    AI[<b>AI health engine</b><br>Korelasi + analisis multi-input]:::blue
    
    Score[<b>Health score</b><br>Skor harian + tren]:::green
    Insight[<b>Insight proaktif</b><br>Pola + prediksi]:::blue
    Companion[<b>AI companion</b><br>Chat kontekstual]:::orange
    
    Dashboard([<b>Dashboard user</b><br>Lihat, eksplor, tanya AI]):::grey

    %% Connections
    Start --> Daily
    
    Daily --> Mood
    Daily --> Sleep
    Daily --> Food
    Daily --> Stress
    
    %% Dashed lines to AI Engine
    Mood -.-> AI
    Sleep -.-> AI
    Food -.-> AI
    Stress -.-> AI
    
    AI --> Score
    AI --> Insight
    AI --> Companion
    
    Score --> Dashboard
    Insight --> Dashboard
    Companion --> Dashboard
    
    %% Feedback loop
    Dashboard -. feedback loop .-> AI
```

---

### ⚙️ Data Layer

```mermaid
graph TD
    %% Styling Classes
    classDef blue fill:#ececff,stroke:#8a8df0,stroke-width:1px,color:#333;
    classDef orange fill:#faebe6,stroke:#d98d75,stroke-width:1px,color:#333;
    classDef yellow fill:#fcefd9,stroke:#d9a86c,stroke-width:1px,color:#333;
    classDef green fill:#e2f5eb,stroke:#5ba889,stroke-width:1px,color:#333;
    classDef grey fill:#efebe9,stroke:#b3a59d,stroke-width:1px,color:#333;
    classDef plain fill:none,stroke:none,color:#333,font-weight:bold;

    %% Layer 1: Input
    NLP[<b>NLP model</b><br>Analisis jurnal]:::blue
    Vision[<b>Vision model</b><br>Deteksi makanan]:::orange
    LSTM[<b>LSTM browser</b><br>Stress dari ketikan]:::yellow
    Sleep[<b>Sleep parser</b><br>Estimasi kualitas tidur]:::green

    %% Layer 2: AI/ML
    Unified[<b>Unified health context store</b><br>Semua input masuk 1 timeline — di-query LLM + model korelasi]:::blue

    %% Layer 3: Data
    Postgres[<b>PostgreSQL</b><br>Events + user data]:::grey
    Chroma[<b>ChromaDB #40;vector#41;</b><br>Memori LLM jangka panjang]:::blue
    Supabase[<b>Supabase storage</b><br>Foto + file media]:::green

    %% Layer 4: Output
    HealthAPI[<b>Health score API</b><br>Skor + breakdown]:::green
    LLMComp[<b>LLM companion</b><br>Chat + rekomendasi]:::blue
    PushNotif[<b>Push notification</b><br>Intervensi proaktif]:::orange

    %% Layer 5: Frontend
    Frontend[Next.js PWA #40;frontend#41;]:::plain

    %% Connections
    NLP --> Unified
    Vision --> Unified
    LSTM --> Unified
    Sleep --> Unified

    Unified --> Postgres
    Unified --> Chroma
    Unified --> Supabase

    Postgres --> HealthAPI
    Chroma --> LLMComp
    Supabase --> PushNotif

    HealthAPI --> Frontend
    LLMComp --> Frontend
    PushNotif --> Frontend
```

### 🎨 Mockup UI

[https://www.figma.com/design/so7jv2beTsQtDXo4PKIq8d/Vitara-UI?node-id=0-1&t=Ofa2OkXLzLm7k5t2-1](https://www.figma.com/design/so7jv2beTsQtDXo4PKIq8d/Vitara-UI?node-id=0-1&t=Ofa2OkXLzLm7k5t2-1)

[Planning — AI Engineer](https://www.notion.so/Planning-AI-Engineer-33a075d3ec7480958bf4ffac0285a9bd?pvs=21)

[ Task Planning — AI Engineer](https://www.notion.so/Task-Planning-AI-Engineer-33c075d3ec74803a91bdecc954a0aeda?pvs=21)