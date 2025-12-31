import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load TRAIN dataset (HC3)
# -----------------------------
train_df = pd.read_csv("hc3_cleaned.csv")

X_train = train_df["clean_text"]
y_train = train_df["label"]

# -----------------------------
# Load TEST dataset (Kaggle)
# -----------------------------
test_df = pd.read_csv("ai_vs_human_kaggle.csv")

# Kaggle column mapping
X_test = test_df["text"]
y_test = test_df["generated"]   # 0 = Human, 1 = AI

# -----------------------------
# TF-IDF (fit on TRAIN only)
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# Train model on HC3
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Test on Kaggle dataset
# -----------------------------
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print("🌍 Cross-Dataset Accuracy:", round(acc, 4))

print("\n📊 Cross-Dataset Classification Report:")
print(classification_report(y_test, y_pred))
