import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load training data
df = pd.read_csv("hc3_cleaned.csv")

X = df["clean_text"]
y = df["label"]

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    lowercase=True
)

X_tfidf = tfidf.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# Save model & vectorizer
joblib.dump(model, "ai_text_detector_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model & Vectorizer saved successfully")
