import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class BBCPreprocessor:
    def __init__(self, articles_dir):
        self.articles_dir = articles_dir
        self.stop_words = set(stopwords.words("english"))
        self.data = []

    def clean_text(self, text):
        """
        Lowercase, remove punctuation, stopwords
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def load_data(self):
        """
        Load BBC news articles into structured list of dicts
        """
        categories = os.listdir(self.articles_dir)
        
        for category in categories:
            category_path = os.path.join(self.articles_dir, category)

            if not os.path.isdir(category_path):
                continue

            for file_name in os.listdir(category_path):
                try:
                    # Load article
                    with open(os.path.join(category_path, file_name), "r", encoding="latin-1") as f:
                        article = f.read()

                    # Clean
                    cleaned_article = self.clean_text(article)
                    sentences = sent_tokenize(article)

                    self.data.append({
                        "category": category,
                        "document": cleaned_article,
                        "sentences": sentences
                    })

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
        
        return self.data

if __name__ == "__main__":
    articles_dir = "data/BBC News Summary/News Articles"

    preprocessor = BBCPreprocessor(articles_dir)
    dataset = preprocessor.load_data()

    print(f"✅ Loaded {len(dataset)} documents")
    print("🔹 Example:")
    print(dataset[0])
