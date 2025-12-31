import re
import pandas as pd

# Load HC3 combined dataset
df = pd.read_csv("hc3_human_vs_ai.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply minimal cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Remove very short/noisy samples
df = df[df["clean_text"].str.len() > 20]

# Save cleaned dataset
df.to_csv("hc3_cleaned.csv", index=False)

print("✅ Preprocessing completed successfully")
print(df["label"].value_counts())
