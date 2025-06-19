# predict_news.py

import pickle

# Load model dan vectorizer
with open("saved_model/model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def predict(news_text):
    processed = vectorizer.transform([news_text])
    prediction = model.predict(processed)
    return "FAKE" if prediction[0] == 1 else "REAL"

# Contoh
if __name__ == "__main__":
    print("Masukkan teks berita:")
    news = input("> ")
    result = predict(news)
    print("🔍 Prediksi:", result)
