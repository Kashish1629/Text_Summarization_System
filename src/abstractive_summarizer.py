from transformers import pipeline

class AbstractiveSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", min_length=30, max_length=120):
        """
        :param model_name: Hugging Face model name (e.g. 'facebook/bart-large-cnn', 't5-small')
        :param min_length: Minimum words in summary
        :param max_length: Maximum words in summary
        """
        print(f"🔄 Loading model {model_name} ...")
        self.summarizer = pipeline("summarization", model=model_name)
        self.min_length = min_length
        self.max_length = max_length
        print("✅ Model loaded successfully!")

    def summarize(self, text: str):
        """
        Generate abstractive summary for given text
        """
        if len(text.strip().split()) < 50:
            return "⚠️ Text too short for summarization."

        summary = self.summarizer(
            text,
            min_length=self.min_length,
            max_length=self.max_length,
            do_sample=False
        )

        return summary[0]["summary_text"]


if __name__ == "__main__":
    doc = """The rapid advancements in artificial intelligence have led to significant
    progress in natural language processing. Applications such as chatbots,
    automated summarization, and translation systems are transforming industries.
    However, challenges such as bias, explainability, and ethical use of AI remain
    areas of active research and concern."""

    # Initialize summarizer with BART
    summarizer = AbstractiveSummarizer(model_name="facebook/bart-large-cnn")
    print("🔹 Abstractive Summary (BART):")
    print(summarizer.summarize(doc))

    # Initialize summarizer with T5
    summarizer_t5 = AbstractiveSummarizer(model_name="t5-small")
    print("\n🔹 Abstractive Summary (T5):")
    print(summarizer_t5.summarize(doc))
