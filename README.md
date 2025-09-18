# 📝 Hybrid Text Summarization System

This project is a **Hybrid Text Summarizer** that generates **Extractive** and **Abstractive** summaries from input text. It leverages classical NLP techniques (TF-IDF, TextRank) for extractive summarization and transformer-based models (BART, T5) for abstractive summarization.

The app also optionally evaluates generated summaries against a reference using **ROUGE metrics**.

---

## 🌟 Features

- **Extractive Summarization**
  - Selects the most important sentences from the input text.
  - Methods: TF-IDF, TextRank.
  - Adjustable number of sentences.

- **Abstractive Summarization**
  - Generates a human-like summary using Hugging Face Transformers.
  - Supported models: BART (`facebook/bart-large-cnn`) and T5 (`t5-small`).

- **Evaluation**
  - Optional reference summary evaluation using ROUGE-1, ROUGE-2, and ROUGE-L scores.

- **Interactive Frontend**
  - Built with Streamlit.
  - Side-by-side display of Extractive and Abstractive summaries.
  - Download summaries as text files.

---

## 🗂 Project Structure

