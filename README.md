# 📰 Fake News Detection using Machine Learning

This project implements a machine learning-based fake news detector that classifies news articles as **FAKE** or **REAL** using Natural Language Processing (NLP). The system is built in Google Colab using Python, and achieves over **96% accuracy** on a balanced dataset.

---

## 📌 Overview

- ✅ Cleans and preprocesses a fake news dataset
- ✅ Combines headline and body text for context
- ✅ Uses TF-IDF vectorization with bi-grams
- ✅ Trains a PassiveAggressiveClassifier
- ✅ Achieves 96%+ accuracy with balanced performance
- ✅ Custom prediction function for live testing

---

## 🧠 Motivation

With the rise of misinformation in digital media, especially on social platforms, there's a critical need to identify and filter out fake news efficiently. This project addresses that need using simple, scalable ML techniques.

---

## 🛠️ Tech Stack

- Python (Google Colab)
- Scikit-learn
- Pandas, NumPy
- TF-IDF Vectorization
- PassiveAggressiveClassifier
- Optional: RandomForestClassifier, LogisticRegression

---

## 📁 Dataset

- Source: [WELFake Dataset](https://www.kaggle.com/datasets/sootersaalu/updated-welfake-dataset)
- Total Articles: ~15,000
- Labels: `REAL`, `FAKE`

---

## ⚙️ How It Works

1. **Data Cleaning**  
   - Removes malformed rows  
   - Standardizes labels (`0`, `1`, `FAKE`, `REAL` → `FAKE`/`REAL`)

2. **Text Preprocessing**  
   - Combines `title` + `text`  
   - Applies TF-IDF vectorization (`ngram_range=(1,2)`)

3. **Model Training**  
   - Balanced dataset using upsampling  
   - PassiveAggressiveClassifier trained on 80% data

4. **Evaluation**  
   - Accuracy: **96.15%**  
   - Precision/Recall/F1: balanced for both FAKE and REAL

---

## 🔍 Sample Predictions

```python
sample = "Aliens have contacted Earth through YouTube signals, claims scientist."
print(detect_fake_news(sample))  # Output: FAKE
