# UTS Machine Learning — Deep Learning vs Classical ML

**Nama:** Daniel Lincoln Ginting  
**NIM:** 1202220201  
**Mata Kuliah:** Machine Learning — Deep Learning  

---

## Deskripsi Proyek

Repository ini berisi implementasi dan perbandingan eksperimental antara metode **Classical Machine Learning** dan **Deep Learning** pada tiga tipe data yang berbeda sebagai bagian dari tugas UTS. Setiap kasus menggunakan dataset publik dari Kaggle, dengan pembagian data, random seed, dan metrik evaluasi yang dijaga konsisten agar perbandingan bersifat fair.

> **Pertanyaan utama:** *Kapan deep learning benar-benar lebih baik dari metode konvensional?*

---

## Struktur Repository

```
├── Kasus1.ipynb        # Titanic Survival Prediction (Data Tabular)
├── Kasus2.ipynb        # Digit Recognizer / MNIST (Data Citra)
├── Kasus3.ipynb        # Disaster Tweet Classification (Data Teks/NLP)
└── README.md
```

---

## Kasus 1 — Titanic Survival Prediction (Data Tabular)

**Notebook:** `Kasus1.ipynb`  
**Dataset:** [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle)

### Deskripsi
Memprediksi apakah seorang penumpang Titanic selamat atau tidak (klasifikasi biner) berdasarkan fitur demografis dan tiket. Dataset training terdiri dari 891 baris dengan 11 fitur asli.

### Pipeline
1. **EDA** — Visualisasi tingkat keselamatan berdasarkan Pclass dan Sex
2. **Preprocessing** — Ekstraksi title dari nama, feature engineering FamilySize, imputasi missing values (Age per-Title, Embarked modus), age binning, one-hot encoding, StandardScaler
3. **Model Konvensional** — Logistic Regression & Random Forest dengan GridSearchCV (5-fold CV)
4. **Deep Learning** — MLP (2 hidden layers: 64→32, BatchNorm, Dropout 0.3, EarlyStopping)
5. **Evaluasi & Analisis Error**

### Hasil

| Model | Test Accuracy | Waktu Training |
|-------|:---:|:---:|
| Logistic Regression | **84.92%** | 2.15 s |
| Random Forest | 83.24% | 23.22 s |
| MLP (Deep Learning) | 79.33% | 6.71 s |

### Kesimpulan Kasus 1
> Pada dataset tabular berukuran kecil (~712 data training), metode konvensional mengungguli deep learning. Logistic Regression dengan parameter terbaik `C=10` mengalahkan MLP dengan selisih ~5.6%. MLP dengan 3.585 parameter mengalami kesulitan generalisasi karena keterbatasan data.

---

## Kasus 2 — Digit Recognizer / MNIST (Data Citra)

**Notebook:** `Kasus2.ipynb`  
**Dataset:** [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) (Kaggle)

### Deskripsi
Mengklasifikasikan gambar digit tulisan tangan 0–9 berukuran 28×28 piksel (grayscale). Dataset terdiri dari 42.000 gambar training dengan distribusi kelas yang seimbang.

### Pipeline
1. **EDA** — Distribusi kelas dan visualisasi contoh gambar per digit
2. **Preprocessing** — Normalisasi pixel (÷255), ekstraksi fitur HOG, reduksi dimensi PCA (95% variance → 153 komponen), reshape untuk CNN `(N, 28, 28, 1)`
3. **Model Konvensional:**
   - HOG + SVM (RBF kernel, C=10, gamma='scale')
   - PCA + Random Forest (n_estimators=100)
4. **Deep Learning** — CNN (Conv2D×2 + MaxPooling + Dropout + Dense, EarlyStopping)
5. **Evaluasi, Confusion Matrix & Analisis Error**

### Arsitektur CNN

```
Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
Conv2D(64, 3×3, ReLU) → MaxPooling(2×2) → Dropout(0.25)
Flatten → Dense(128, ReLU) → Dropout(0.5)
Dense(10, Softmax)
Total Parameter: 225.034
```

### Hasil

| Model | Test Accuracy | Waktu Training |
|-------|:---:|:---:|
| HOG + SVM | 97.04% | 14.34 s |
| PCA + Random Forest | 94.13% | 78.94 s |
| CNN (Deep Learning) | **98.99%** | 291.61 s |

### Kesimpulan Kasus 2
> CNN berhasil mengungguli metode konvensional pada data citra. Kemampuan convolutional layer untuk belajar fitur spasial hierarkis secara end-to-end (tanpa feature engineering manual seperti HOG) menjadi keunggulan utamanya. Peningkatan ini datang dengan trade-off waktu training yang jauh lebih lama (~20× dibanding HOG+SVM).

---

## Kasus 3 — Natural Language Processing with Disaster Tweets (Data Teks)

**Notebook:** `Kasus3.ipynb`  
**Dataset:** [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) (Kaggle)

### Deskripsi
Mengklasifikasikan tweet berbahasa Inggris sebagai bencana nyata (1) atau bukan/metaforis (0). Dataset terdiri dari 7.613 tweet dengan distribusi kelas yang cukup seimbang (~57% non-bencana, ~43% bencana). Metrik evaluasi utama adalah **F1-Score** karena adanya class imbalance ringan.

### Pipeline
1. **EDA** — Distribusi kelas target dan preview tweet mentah
2. **Preprocessing Teks** — Lowercasing, hapus URL, hapus mention/hashtag symbol, hapus karakter non-alfabet
3. **Model Konvensional:**
   - TF-IDF (unigram+bigram, max 10.000 fitur) + Logistic Regression
   - TF-IDF + Multinomial Naive Bayes
4. **Deep Learning** — BiLSTM (Embedding 64-dim → Bidirectional LSTM(64) → Dense(32) → Sigmoid, EarlyStopping)
5. **Evaluasi & Analisis Error dengan contoh tweet**

### Arsitektur BiLSTM

```
Embedding(vocab=10.000, dim=64, max_len=30)
Bidirectional(LSTM(64)) → Dropout(0.5)
Dense(32, ReLU) → Dropout(0.5)
Dense(1, Sigmoid)
Total Parameter: 710.209
```

### Hasil

| Model | F1-Score (Validation) | Waktu Training |
|-------|:---:|:---:|
| Logistic Regression + TF-IDF | **76.39%** | 0.054 s |
| Naive Bayes + TF-IDF | 75.35% | 0.005 s |
| BiLSTM (Deep Learning) | 75.57% | 37.05 s |

### Kesimpulan Kasus 3
> BiLSTM yang dilatih dari scratch tidak mampu melampaui TF-IDF + Logistic Regression pada dataset ini. TF-IDF dengan bigram efektif menangkap frasa-frasa khas bencana, sementara BiLSTM tanpa pretrained embeddings belum memiliki pemahaman semantik yang cukup dalam. Penggunaan pretrained Transformer (BERT/DistilBERT) kemungkinan akan mengubah hasil ini secara signifikan.

---

## Rangkuman Komparatif

| Kasus | Tipe Data | Pemenang | Model Terbaik | Skor |
|-------|-----------|:---:|---|:---:|
| Titanic | Tabular (kecil) | ✅ Classical | Logistic Regression | 84.92% acc |
| MNIST | Citra | ✅ Deep Learning | CNN | 98.99% acc |
| Disaster Tweets | Teks (NLP) | ✅ Classical | LR + TF-IDF | 76.39% F1 |

### Kapan Deep Learning Lebih Unggul?
- Data berukuran **besar**
- Data memiliki **struktur spasial** (gambar → CNN) atau **sekuensial** (teks panjang → LSTM/Transformer)
- Fitur sulit didefinisikan secara manual → end-to-end learning lebih menguntungkan

### Kapan Classical ML Masih Kompetitif?
- Dataset **kecil** (< beberapa ribu sampel)
- Data **tabular** dengan fitur yang dapat di-engineer secara manual
- Kebutuhan akan **interpretabilitas** tinggi
- Keterbatasan **sumber daya komputasi**

---

## Cara Menjalankan

Semua notebook dirancang untuk dijalankan di **Google Colab**.

### 1. Buka notebook di Colab
Klik badge di bawah atau upload file `.ipynb` secara manual ke [colab.research.google.com](https://colab.research.google.com).

### 2. Upload dataset yang diperlukan

| Notebook | File Dataset yang Dibutuhkan |
|----------|------------------------------|
| `Kasus1.ipynb` | `train.csv` dari [Titanic Kaggle](https://www.kaggle.com/c/titanic/data) |
| `Kasus2.ipynb` | `train-kasus2.csv` dari [Digit Recognizer Kaggle](https://www.kaggle.com/c/digit-recognizer/data) |
| `Kasus3.ipynb` | `train-kasus3.csv` dari [Disaster Tweets Kaggle](https://www.kaggle.com/c/nlp-getting-started/data) |

> Rename file dataset sesuai nama di atas sebelum diupload ke sesi Colab.

### 3. Jalankan semua cell secara berurutan
`Runtime → Run all` atau jalankan tiap cell dengan `Shift+Enter`.

> Untuk Kasus 2 (CNN), disarankan menggunakan **GPU runtime**: `Runtime → Change runtime type → GPU`

---

## Dependencies

```
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
scikit-image       # HOG feature extraction (Kasus 2)
tensorflow >= 2.x  # MLP, CNN, BiLSTM
```

Semua library di atas tersedia secara default di Google Colab dan tidak perlu instalasi tambahan.

---

## Random Seed

Seluruh eksperimen menggunakan **random seed = 42** secara konsisten (Python `random`, NumPy, TensorFlow) untuk menjamin reproducibility hasil.

---