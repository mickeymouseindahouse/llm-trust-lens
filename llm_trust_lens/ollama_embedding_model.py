from typing import List, Optional
import requests
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from llm_trust_lens.embedding_model_interface import EmbeddingModelInterface


class OllamaEmbeddingModel(EmbeddingModelInterface):
    def __init__(
        self,
        model_name: str = "llama3.2",
        api_url: str = "http://localhost:11434",
        dataset_name: str = "uclanlp/wino_bias",
        dataset_config: str = "type1_anti",
        text_column: str = "tokens",
    ):
        """
        Initialize the Ollama embedding model.
        
        Args:
            model_name: Name of the Ollama model to use
            api_url: URL of the Ollama API
            dataset_name: Name of the Hugging Face dataset to load
            dataset_config: Configuration name for the dataset
            text_column: Name of the column containing text in the dataset
        """
        self.model_name = model_name
        self.api_url = api_url
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, dataset_config, split="validation")
        
        # For WinoBias dataset, tokens are lists, so we need to join them
        if text_column == "tokens" and isinstance(self.dataset[0][text_column], list):
            self.texts = [" ".join(item[text_column]) for item in self.dataset]
        else:
            self.texts = self.dataset[text_column]
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text input using Ollama API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        url = f"{self.api_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding from response
            if "embedding" in result:
                return result["embedding"]
            else:
                raise ValueError(f"No embedding found in response: {result}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding from Ollama: {e}")
            return [0.0]
    def retrieve_similar(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve similar texts from the loaded dataset.
        
        Args:
            query: Query text
            top_k: Number of similar texts to retrieve
            
        Returns:
            List of similar texts
        """
        query_embedding = np.array(self.get_embedding(query))
        
        text_embeddings = []
        for text in tqdm(self.texts):
            text_embedding = np.array(self.get_embedding(text))
            text_embeddings.append(text_embedding)
        
        similarities = []
        for text_embedding in text_embeddings:
            norm_query = query_embedding / np.linalg.norm(query_embedding)
            norm_text = text_embedding / np.linalg.norm(text_embedding)
            similarity = np.dot(norm_query, norm_text)
            similarities.append(similarity)
        
        top_indices = sorted(range(len(similarities)),
                           key=lambda i: similarities[i], 
                           reverse=True)[:top_k]
        
        return [self.texts[i] for i in top_indices]
