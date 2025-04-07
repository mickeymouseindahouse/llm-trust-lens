from typing import List, Dict, Any
from .embedding_model_interface import EmbeddingModelInterface
from .trust_evaluators import BiasEvaluator, RetrievalEvaluator, HallucinationEvaluator

class TrustEvaluator:
    def __init__(self, model: EmbeddingModelInterface):
        self.model = model
        self.bias_evaluator = BiasEvaluator(model)
        self.retrieval_evaluator = RetrievalEvaluator(model)
        self.hallucination_evaluator = HallucinationEvaluator(model)

    def evaluate(self, 
                trust_metrics: List[str],
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate trust metrics for the model.
        
        Args:
            trust_metrics: List of metrics to evaluate ("bias", "retrieval", "hallucination")
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {}
        
        if "bias" in trust_metrics:
            results["bias"] = self.bias_evaluator.evaluate(
                demographic_pairs=kwargs.get("demographic_pairs", []),
                attribute_words=kwargs.get("attribute_words", [])
            )
            
        if "retrieval" in trust_metrics:
            results["retrieval"] = self.retrieval_evaluator.evaluate(
                queries=kwargs.get("queries", []),
                ground_truth=kwargs.get("ground_truth", []),
                top_k=kwargs.get("top_k", 5)
            )
            
        if "hallucination" in trust_metrics:
            results["hallucination"] = self.hallucination_evaluator.evaluate(
                claims=kwargs.get("claims", []),
                evidence=kwargs.get("evidence", []),
                threshold=kwargs.get("threshold", 0.7)
            )
            
        return results

    def plot_results(self, metric: str):
        """
        Plot evaluation results for a specific metric.
        
        Args:
            metric: Metric to plot ("bias", "retrieval", "hallucination")
        """
        # TODO: Implement plotting functionality
        pass
