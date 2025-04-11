import streamlit as st
from newsapi import NewsApiClient
import re, string, pickle
from nltk.corpus import stopwords
import numpy as np
from textblob import TextBlob

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

st.title("📰 Fake News Detector")

# Mode selection
mode = st.radio("Choose Input Mode:", ["📝 Paste News Text", "🛰️ Use Live News"])

if mode == "📝 Paste News Text":
    user_input = st.text_area("Paste your news article here")

elif mode == "🛰️ Use Live News":
    newsapi = NewsApiClient(api_key='1d2b5ee3098a4cdf88d81d30bb91cc92')  # Replace with your API key
    top_headlines = newsapi.get_top_headlines(language='en', country='us', page_size=20)
    options = [article['title'] for article in top_headlines['articles']]
    user_input = st.selectbox("Choose a live news headline", options)

# Check button
if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter or select a news text.")
    else:
        clean = clean_text(user_input)
        length = len(clean)
        uppercase = sum(1 for c in clean if c.isupper())
        sentiment = TextBlob(clean).sentiment.polarity

        X_extra = np.array([[length, uppercase, sentiment]])
        X_tfidf = tfidf.transform([clean])
        final_input = np.hstack([X_tfidf.toarray(), X_extra])

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            st.error("🚨 This news might be *fake*.")
        else:
            st.success("✅ This news appears to be *real*.")
