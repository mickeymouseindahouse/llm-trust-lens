from typing import List, Dict, Any
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import precision_recall_fscore_support
from .embedding_model_interface import EmbeddingModelInterface

class BiasEvaluator:
    def __init__(self, model: EmbeddingModelInterface):
        self.model = model
        
    def evaluate(self, demographic_pairs: List[tuple[str, str]], 
                attribute_words: List[str]) -> Dict[str, float]:
        """
        Evaluate bias in embeddings using WEAT (Word Embedding Association Test).
        
        Args:
            demographic_pairs: List of (target1, target2) word pairs (e.g., [("man", "woman")])
            attribute_words: List of attribute words to test bias against
            
        Returns:
            Dictionary containing bias scores and p-values
        """
        results = {}
        
        for target1, target2 in demographic_pairs:
            # Get embeddings for target words
            target1_emb = np.array([self.model.get_embedding(word) for word in target1])
            target2_emb = np.array([self.model.get_embedding(word) for word in target2])
            
            # Get embeddings for attribute words

            # engineer
            attr_emb = np.array([self.model.get_embedding(word) for word in attribute_words])
            
            # Calculate mean embeddings
            target1_mean = np.mean(target1_emb, axis=0)
            target2_mean = np.mean(target2_emb, axis=0)
            attr_mean = np.mean(attr_emb, axis=0)

            # he, man -- she, woman
            #
            
            # Calculate cosine similarities
            target1_sim = np.dot(target1_mean, attr_mean) / (np.linalg.norm(target1_mean) * np.linalg.norm(attr_mean))
            target2_sim = np.dot(target2_mean, attr_mean) / (np.linalg.norm(target2_mean) * np.linalg.norm(attr_mean))
            
            # Calculate effect size and p-value
            effect_size = target1_sim - target2_sim
            _, p_value = ttest_ind(target1_emb, target2_emb)
            
            results[f"{target1[0]}_vs_{target2[0]}"] = {
                "effect_size": effect_size,
                "p_value": p_value
            }
            
        return results

class RetrievalEvaluator:
    def __init__(self, model: EmbeddingModelInterface):
        self.model = model
        
    def evaluate(self, queries: List[str], 
                ground_truth: List[List[str]],
                top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval effectiveness using precision, recall, and F1.
        
        Args:
            queries: List of query texts
            ground_truth: List of lists containing relevant documents for each query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        all_precision = []
        all_recall = []
        all_f1 = []
        
        for query, relevant_docs in zip(queries, ground_truth):
            # Get retrieved documents
            retrieved = self.model.retrieve_similar(query, top_k=top_k)
            
            # Calculate metrics
            relevant_retrieved = set(retrieved) & set(relevant_docs)
            precision = len(relevant_retrieved) / len(retrieved)
            recall = len(relevant_retrieved) / len(relevant_docs)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
        return {
            "precision": np.mean(all_precision),
            "recall": np.mean(all_recall),
            "f1": np.mean(all_f1)
        }

class HallucinationEvaluator:
    def __init__(self, model: EmbeddingModelInterface):
        self.model = model
        
    def evaluate(self, 
                claims: List[str],
                evidence: List[str],
                threshold: float = 0.7) -> Dict[str, float]:
        """
        Evaluate hallucination by comparing claim embeddings with evidence embeddings.
        
        Args:
            claims: List of claims to evaluate
            evidence: List of evidence texts
            threshold: Similarity threshold for considering a claim supported
            
        Returns:
            Dictionary containing hallucination scores
        """
        hallucination_scores = []
        
        for claim in claims:
            # Get claim embedding
            claim_emb = np.array(self.model.get_embedding(claim))
            
            # Get evidence embeddings
            evidence_embs = np.array([self.model.get_embedding(text) for text in evidence])
            
            # Calculate similarities
            similarities = np.dot(evidence_embs, claim_emb) / (
                np.linalg.norm(evidence_embs, axis=1) * np.linalg.norm(claim_emb)
            )
            
            # Calculate hallucination score (1 - max similarity)
            max_similarity = np.max(similarities)
            hallucination_score = 1 - max_similarity if max_similarity < threshold else 0
            hallucination_scores.append(hallucination_score)
            
        return {
            "hallucination_rate": np.mean(hallucination_scores),
            "max_hallucination": np.max(hallucination_scores)
        } 