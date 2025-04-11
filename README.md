# 📰 Fake News Detector (ML + Streamlit)

Hey there! This is a simple project I built to detect whether a news article is real or fake using machine learning. It runs on **Streamlit**, and you can either **paste your own article** or pull **live headlines** using NewsAPI.

## 🔧 What it Does

- You can paste any news article or sentence and it’ll tell you if it’s likely fake or real.
- It also has a “fetch live news” option to test the model on real headlines.
- Behind the scenes:
  - TF-IDF vectorizer (1–2 grams, top 5000 words)
  - Some smart features like article length, uppercase word count, and sentiment
  - Trained with Logistic Regression (simple but works well!)

## 🗂️ Files

- `app.py` → Streamlit app
- `model.pkl` → Trained ML model
- `tfidf_vectorizer.pkl` → Vectorizer used to process input text
- `requirements.txt` → Packages you’ll need
- `README.md` → This file!

## Training:

- Dataset: News headlines + custom preprocessing
- Cleaned text, added basic NLP features
- Vectorized using `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`
- Added extra features like:
  - Text length
  - Uppercase count
  - Sentiment score
- Final model trained with `LogisticRegression`

## Run:

1. Clone the repo:
   ```bash
   git clone https://github.com/sreejabavandla/FakeNewsDetection/
   cd fake-news-detector
