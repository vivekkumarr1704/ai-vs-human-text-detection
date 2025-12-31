import os
import json
import pandas as pd

DATA_DIR = "hc3_raw"

human_rows = []
ai_rows = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".jsonl"):
        file_path = os.path.join(DATA_DIR, file)
        print(f"Processing {file}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                if "human_answers" in item:
                    for h in item["human_answers"]:
                        human_rows.append({
                            "text": h,
                            "label": 0,
                            "source": file.replace(".jsonl", "")
                        })

                if "chatgpt_answers" in item:
                    for a in item["chatgpt_answers"]:
                        ai_rows.append({
                            "text": a,
                            "label": 1,
                            "source": file.replace(".jsonl", "")
                        })

df = pd.DataFrame(human_rows + ai_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("hc3_human_vs_ai.csv", index=False)

print("✅ HC3 JSONL successfully converted to CSV")
print(df["label"].value_counts())
