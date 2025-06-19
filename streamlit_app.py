# streamlit_app.py

import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.markdown("Masukkan sebuah berita dan sistem akan memprediksi apakah berita tersebut **REAL** atau **FAKE**.")

text_input = st.text_area("🧾 Tulis atau tempel berita di bawah ini:")

if st.button("🔍 Deteksi"):
    if text_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        input_vec = vectorizer.transform([text_input])
        prediction = model.predict(input_vec)[0]
        label = "✅ REAL" if prediction == 0 else "❌ FAKE"
        st.success(f"Hasil Deteksi: **{label}**")
