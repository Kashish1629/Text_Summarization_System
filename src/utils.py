import re
import nltk
from nltk.corpus import stopwords

# Make sure required NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercasing
    - Removing special characters, numbers
    - Removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens
    """
    return [word for word in tokens if word not in STOPWORDS]

def tokenize_sentences(text: str):
    """
    Sentence tokenization
    """
    return nltk.sent_tokenize(text)

def tokenize_words(text: str):
    """
    Word tokenization
    """
    return nltk.word_tokenize(text)

def load_text_file(file_path: str) -> str:
    """
    Utility to load text data from a file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_text_file(file_path: str, text: str):
    """
    Save processed text into a file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
