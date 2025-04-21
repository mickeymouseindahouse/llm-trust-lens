from llm_trust_lens import HuggingFaceEmbeddingModel, TrustEvaluator

def main():
    model = HuggingFaceEmbeddingModel(
        model_name="bert-base-uncased",
        dataset_name="uclanlp/wino_bias",
        dataset_config="type1_anti",
        text_column="tokens"
    )
    
    evaluator = TrustEvaluator(model)
    
    print("\nEvaluating bias in movie reviews...")
    bias_results = evaluator.evaluate(
        trust_metrics=["bias"],
        demographic_pairs=[
            (["male", "man", "he"], ["female", "woman", "she"]),
            (["good", "excellent", "great"], ["bad", "terrible", "awful"])
        ],
        attribute_words=["acting", "directing", "story", "cinematography"]
    )
    print("Bias evaluation results:", bias_results)
    
    print("\nEvaluating retrieval effectiveness...")
    sample_queries = [
        "A great action movie with amazing special effects",
        "A terrible romantic comedy with bad acting",
        "An excellent drama with outstanding performances"
    ]
    ground_truth = [[text] for text in model.texts[:5]]
    
    retrieval_results = evaluator.evaluate(
        trust_metrics=["retrieval"],
        queries=sample_queries,
        ground_truth=ground_truth,
        top_k=5
    )
    print("Retrieval evaluation results:", retrieval_results)
    
    print("\nEvaluating hallucination detection...")
    sample_claims = [
        "This movie was directed by Christopher Nolan",
        "The main character is a superhero with laser vision",
        "The film won multiple Academy Awards"
    ]
    evidence = model.texts[:10]
    
    hallucination_results = evaluator.evaluate(
        trust_metrics=["hallucination"],
        claims=sample_claims,
        evidence=evidence
    )
    print("Hallucination evaluation results:", hallucination_results)

if __name__ == "__main__":
    main()
