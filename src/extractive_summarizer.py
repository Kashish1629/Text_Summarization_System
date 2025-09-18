import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


class ExtractiveSummarizer:
    def __init__(self, method="tfidf", num_sentences=3):
        """
        :param method: 'tfidf' or 'textrank'
        :param num_sentences: number of sentences in summary
        """
        self.method = method
        self.num_sentences = num_sentences

    def summarize(self, document: str):
        """
        Generate extractive summary using selected method
        """
        sentences = sent_tokenize(document)
        if len(sentences) <= self.num_sentences:
            return document  # already short

        if self.method == "tfidf":
            return self._summarize_tfidf(sentences)
        elif self.method == "textrank":
            return self._summarize_textrank(sentences)
        else:
            raise ValueError("Method must be 'tfidf' or 'textrank'")

    def _summarize_tfidf(self, sentences):
        """
        TF-IDF scoring: pick sentences with highest average TF-IDF score
        """
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # Sentence importance = mean TF-IDF value
        sentence_scores = np.array(X.mean(axis=1)).ravel()

        # Select top-k sentences
        top_indices = sentence_scores.argsort()[-self.num_sentences:][::-1]

        summary = [sentences[i] for i in sorted(top_indices)]
        return " ".join(summary)

    def _summarize_textrank(self, sentences):
        """
        TextRank: graph-based ranking of sentences
        """
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # Cosine similarity matrix
        sim_matrix = cosine_similarity(X)

        # Normalize
        np.fill_diagonal(sim_matrix, 0)
        scores = np.sum(sim_matrix, axis=1)

        # Pick top-k sentences
        top_indices = scores.argsort()[-self.num_sentences:][::-1]
        summary = [sentences[i] for i in sorted(top_indices)]
        return " ".join(summary)
