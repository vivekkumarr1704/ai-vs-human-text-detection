from datasets import load_dataset
import pandas as pd

print("Loading HC3 dataset...")

dataset = load_dataset(
    "Hello-SimpleAI/HC3",
    trust_remote_code=True
)

human = []
ai = []

for item in dataset["train"]:
    for h in item["human_answers"]:
        human.append({"text": h, "label": 0})
    for a in item["chatgpt_answers"]:
        ai.append({"text": a, "label": 1})

df = pd.DataFrame(human + ai)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("hc3_human_vs_ai.csv", index=False)

print("✅ HC3 CSV created successfully!")
print(df.head())
print(df["label"].value_counts())
