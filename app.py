import streamlit as st
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.evaluation import Evaluator

st.set_page_config(page_title="Hybrid Text Summarizer", layout="wide")

# Title and description
st.title("📝 TEXT SUMMARIZATION SYSTEM")
st.markdown(
    """
    This app generates **Extractive** and **Abstractive** summaries for your text.
    
    - **Extractive**: Selects the most important sentences (TF-IDF/TextRank)
    - **Abstractive**: Generates a human-like summary (BART/T5)
    """
)

# Sidebar options
st.sidebar.header("Settings")
num_sentences = st.sidebar.slider("Number of sentences (Extractive)", 1, 10, 3)
extractive_method = st.sidebar.selectbox("Extractive Method", ["tfidf", "textrank"])
abstractive_model = st.sidebar.selectbox("Abstractive Model", ["facebook/bart-large-cnn", "t5-small"])

# Main text input
input_text = st.text_area("Paste your article/text here:", height=250)

# Generate summary button
if st.button("Generate Summaries"):
    if not input_text.strip():
        st.warning("Please paste some text to summarize!")
    else:
        # Initialize summarizers
        extractive_summarizer = ExtractiveSummarizer(method=extractive_method, num_sentences=num_sentences)
        abstractive_summarizer = AbstractiveSummarizer(model_name=abstractive_model)

        # Generate summaries
        extractive_summary = extractive_summarizer.summarize(input_text)
        abstractive_summary = abstractive_summarizer.summarize(input_text)

        # Display summaries side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 Extractive Summary")
            st.write(extractive_summary)
            st.download_button("📥 Download Extractive Summary", extractive_summary, file_name="extractive_summary.txt")

        with col2:
            st.subheader("🔹 Abstractive Summary")
            st.write(abstractive_summary)
            st.download_button("📥 Download Abstractive Summary", abstractive_summary, file_name="abstractive_summary.txt")

        st.success("✅ Summarization complete!")

        
