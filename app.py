import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader

# --------------------------------
# Load model & vectorizer
# --------------------------------
model = joblib.load("ai_text_detector_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# --------------------------------
# Clean text
# --------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------
# Extract text from PDF
# --------------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="AI vs Human Text Detector", layout="centered")

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🧠 AI vs Human Detection", "📊 Model Evaluation"]
)

# --------------------------------
# Dark Mode Toggle
# --------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    bg = "background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color:white;"
    card_bg = "#1e293b"
else:
    bg = "background: linear-gradient(135deg, #e3f2fd, #ffffff); color:black;"
    card_bg = "white"

st.markdown(
    f"""
    <style>
    .stApp {{ {bg} }}
    .card {{
        background-color: {card_bg};
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin-top: 20px;
    }}
    .human {{ border-left: 8px solid #22c55e; }}
    .ai {{ border-left: 8px solid #ef4444; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PAGE 1: AI vs Human Detection
# =========================================================
if page == "🧠 AI vs Human Detection":

    st.markdown("<h1 style='text-align:center;'>🧠 AI vs Human Text Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:18px;'>AI vs Human Text Detection System</p>", unsafe_allow_html=True)

    # Input Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose input method:",
        ["✍️ Paste Text", "📁 Upload File (PDF/TXT)"]
    )

    text_input = ""

    if input_mode == "✍️ Paste Text":
        text_input = st.text_area("Paste your text:", height=220)

    else:
        uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text_input = extract_pdf_text(uploaded_file)
            else:
                text_input = uploaded_file.read().decode("utf-8")

    st.markdown("</div>", unsafe_allow_html=True)

    # Analyze Button
    if st.button("🔍 Analyze Text"):
        if len(text_input.strip()) < 20:
            st.warning("⚠️ Text too short for analysis.")
        else:
            cleaned = clean_text(text_input)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            words = len(cleaned.split())
            chars = len(cleaned)

            if prediction == 0:
                label = "Human-written"
                confidence = prob[0] * 100
                risk = "Low"
                cls = "human"
            else:
                label = "AI-generated"
                confidence = prob[1] * 100
                risk = "High" if confidence > 85 else "Medium"
                cls = "ai"

            st.markdown("## 📄 Detection Report")

            st.markdown(
                f"""
                <div class="card {cls}">
                    <h3>{label}</h3>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                    <p><b>Risk Level:</b> {risk}</p>
                    <p><b>Word Count:</b> {words}</p>
                    <p><b>Character Count:</b> {chars}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int(confidence))

# =========================================================
# PAGE 2: Model Evaluation
# =========================================================
if page == "📊 Model Evaluation":

    st.markdown("<h1 style='text-align:center;'>📊 Model Evaluation Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("""
    This section presents the **quantitative evaluation** of the proposed  
    **AI vs Human Text Detection System** using standard performance metrics.
    """)

    # Load evaluation CSV
    eval_df = pd.read_csv("evaluation_results.csv")

    st.subheader("📋 Evaluation Metrics Table")
    st.dataframe(eval_df, use_container_width=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for metric in metrics:
        st.subheader(f"📈 {metric} Comparison")

        fig, ax = plt.subplots()
        ax.bar(eval_df["Model"], eval_df[metric])
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=15)

        st.pyplot(fig)

    st.success("✔ Model evaluation loaded successfully")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:14px;'>AI vs Human Text Detection</p>",
    unsafe_allow_html=True
)
