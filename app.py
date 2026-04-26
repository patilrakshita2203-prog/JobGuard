import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from models.predictor import JobPredictor
# Load model
clf = joblib.load("models/fake_job_model.pkl")

# Load BERT model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("JobGuard - Fake Job Detection System")

st.write("Paste a job description below to check whether it may be Fake or Real.")

job_desc = st.text_area("Enter Job Description")

if st.button("Predict"):
    if job_desc.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Convert text into embedding first
        embedding = bert_model.encode([job_desc])

        # Predict using trained model
        prediction = clf.predict(embedding)[0]
        probability = clf.predict_proba(embedding)[0]

        if prediction == 1:
            st.error(f"⚠ Fake Job Detected! Confidence: {max(probability)*100:.2f}%")
        else:
            st.success(f"✅ Real Job Posting! Confidence: {max(probability)*100:.2f}%")