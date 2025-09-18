from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self, metrics=["rouge1", "rouge2", "rougeL"]):
        """
        :param metrics: list of ROUGE metrics to compute
        """
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    def evaluate_summary(self, reference: str, generated: str):
        """
        Compute ROUGE scores between reference and generated summary
        :param reference: ground truth summary
        :param generated: model/system generated summary
        :return: dict of scores
        """
        scores = self.scorer.score(reference, generated)
        return {metric: round(scores[metric].fmeasure, 4) for metric in scores}

    def evaluate_batch(self, references, generated_summaries):
        """
        Evaluate multiple (ref, gen) pairs
        :param references: list of ground truth summaries
        :param generated_summaries: list of system summaries
        :return: average scores across dataset
        """
        assert len(references) == len(generated_summaries), "Mismatch in lengths!"
        
        results = {m: [] for m in self.metrics}

        for ref, gen in zip(references, generated_summaries):
            scores = self.evaluate_summary(ref, gen)
            for m, v in scores.items():
                results[m].append(v)

        # Average across all samples
        avg_scores = {m: round(sum(vals) / len(vals), 4) for m, vals in results.items()}
        return avg_scores


if __name__ == "__main__":
    # Example Usage
    reference = "AI is transforming industries but ethical challenges remain."
    generated = "Artificial intelligence is changing industries, but ethics are a concern."

    evaluator = Evaluator()
    print("🔹 Single Example Evaluation:")
    print(evaluator.evaluate_summary(reference, generated))

    refs = [
        "AI is transforming industries but ethical challenges remain.",
        "Oil prices rose today due to increasing global demand."
    ]
    gens = [
        "Artificial intelligence is changing industries, but ethics are a concern.",
        "Global demand caused oil prices to increase today."
    ]

    print("\n🔹 Batch Evaluation:")
    print(evaluator.evaluate_batch(refs, gens))
