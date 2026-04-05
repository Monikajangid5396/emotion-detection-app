import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
import re
import pickle

# Ensure stopwords are available
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load saved models/vectorizer
model = pickle.load(open('logistic_regresion.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def prediction(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    probabilities = model.predict_proba(input_vectorized)[0]
    confidence = np.max(probabilities)

    return predicted_emotion, predicted_label, confidence, probabilities

# Streamlit UI
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

# Custom header
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Emotion Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Analyze text and predict one of six emotions: **Joy, Fear, Love, Anger, Sadness, Surprise**.")

# Input section
st.markdown("### ✍️ Enter text to analyze")
input_text = st.text_area("", height=150, placeholder="Type or paste your sentence here...")

# Prediction
if st.button("🔍 Analyze Emotion"):
    if input_text.strip():
        predicted_emotion, predicted_label, confidence, probabilities = prediction(input_text)

        # Results layout
        col1, col2 = st.columns([2,1])

        with col1:
            st.markdown("#### 🎭 Predicted Emotion")
            st.markdown(f"<h2 style='color:#27AE60;'>{predicted_emotion}</h2>", unsafe_allow_html=True)
            st.metric(label="Confidence (%)", value=f"{confidence*100:.2f}")

        with col2:
            st.markdown("#### 📊 Label ID")
            st.info(predicted_label)

        st.markdown("---")
        st.markdown("### Probability Distribution")
        emotions = lb.classes_
        prob_df = pd.DataFrame({"Emotion": emotions, "Probability": probabilities})
        st.bar_chart(prob_df.set_index("Emotion"))

    else:
        st.warning("⚠️ Please enter some text before analyzing.")
