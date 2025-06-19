import pandas as pd
import joblib
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Baca dataset
file_path = "dataset/fake_news_dataset.csv"  # Ganti jika nama file berbeda
df = pd.read_csv(file_path)
print("✅ Dataset berhasil dibaca")

# Hapus baris kosong
df = df.dropna()

# Normalisasi label ke huruf besar (jaga-jaga)
df['label'] = df['label'].astype(str).str.upper()
print("Label unik setelah upper():", df['label'].unique())

# Filter hanya yang label-nya FAKE atau REAL
df = df[df['label'].isin(['FAKE', 'REAL'])]

# Konversi label ke angka
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
print("Label unik setelah mapping:", df['label'].unique())

# Gabungkan title + text menjadi content
df['content'] = df['title'] + ' ' + df['text']

# Preprocessing ringan
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['content'] = df['content'].apply(clean_text)

# Cek ulang jumlah data
print("Jumlah data setelah bersih:", len(df))

# Pisahkan data
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Latih model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluasi
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi model: {acc:.2f}")

# Simpan model dan vectorizer
os.makedirs("saved_model", exist_ok=True)
with open("saved_model/model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("📁 Model berhasil disimpan di: saved_model/model.pkl")
joblib.dump(model, "saved_model/model.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")  # ⬅ WAJIB

