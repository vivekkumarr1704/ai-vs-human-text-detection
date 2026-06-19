# 🧠 AI vs Human Text Detection

A Machine Learning–based system to detect whether a given text is **Human-written** or **AI-generated**.  
This project uses classical NLP techniques and provides an interactive **Streamlit web interface** for easy testing.

---

## 📌 Project Overview

With the rapid growth of AI-generated content, distinguishing between human-written and AI-generated text has become increasingly important.  
This project addresses this challenge by building a complete pipeline that includes:

- Dataset collection
- Text preprocessing
- Feature extraction
- Model training and evaluation
- Interactive web-based deployment

---

## 📂 Project Structure

AI VS HUMAN DETECTION/
│
├── app.py # Streamlit web application
├── requirements.txt # Project dependencies
├── background.jpeg # UI background image
│
├── preprocess_text.py # Text cleaning & preprocessing
├── train_baseline_model.py # Model training script
├── train_compare_models.py # Model comparison script
├── save_model.py # Save trained model
│
├── ai_text_detector_model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── evaluation_results.csv # Evaluation metrics
├── model_comparison_results.csv # Accuracy, F1-score comparison
│
├── download_hc3.py # HC3 dataset download script
├── download_hc3_repo.py # HC3 repository fetch
├── hc3_jsonl_to_csv.py # JSONL → CSV conversion
├── cross_dataset_test.py # Cross-dataset evaluation
│
├── .gitignore # Ignored files (datasets, venv, cache)
└── README.md # Project documentation

---

## 📊 Datasets Used

1. **HC3 (Human–ChatGPT Comparison) Dataset**  
   Source: https://huggingface.co/datasets/Hello-SimpleAI/HC3  
   Domains:
   - Wikipedia
   - Reddit (ELI5)
   - Medicine
   - Finance
   - Open-QA

2. **AI vs Human Text Dataset (Kaggle)**  
   Source: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

> ⚠️ Large datasets are **not uploaded to GitHub** due to size limits.

---

## ⚙️ Technologies Used

- **Python**
- **Scikit-learn**
- **TF-IDF Vectorization**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Streamlit**
- **Joblib**
- **PyPDF2**

---

## 🧪 Machine Learning Pipeline

1. Data collection from public datasets  
2. Text preprocessing (cleaning, normalization)  
3. Feature extraction using TF-IDF  
4. Model training using classical ML classifiers  
5. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
6. Saving best-performing model  
7. Deployment using Streamlit UI  

---

## 🚀 How to Run the Project

### Step 1: Clone Repository
```bash
git clone https://github.com/RistikaSingh/ai-vs-human-text-detection.git
cd ai-vs-human-text-detection
Step 2: Install Dependencies
pip install -r requirements.txt


(If requirements.txt causes issues)

pip install streamlit joblib scikit-learn matplotlib pandas PyPDF2 seaborn

Step 3: Run Application
streamlit run app.py

Step 4: Open Browser
http://localhost:8501