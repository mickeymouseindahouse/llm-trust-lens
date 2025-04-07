from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from llm_trust_lens.embedding_model_interface import EmbeddingModelInterface
from tqdm import tqdm


class HuggingFaceEmbeddingModel(EmbeddingModelInterface):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dataset_name: str = "imdb",
        text_column: str = "text",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the Hugging Face embedding model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            dataset_name: Name of the Hugging Face dataset to load
            text_column: Name of the column containing text in the dataset
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, split="train")
        self.texts = self.dataset[text_column]
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text input.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get model output
            outputs = self.model(**inputs)
            
            # Use the [CLS] token embedding or mean pooling
            if hasattr(outputs, "last_hidden_state"):
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            else:
                embedding = outputs.pooler_output.squeeze().cpu().numpy()
                
            return embedding.tolist()
    
    def retrieve_similar(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve similar texts from the loaded dataset.
        
        Args:
            query: Query text
            top_k: Number of similar texts to retrieve
            
        Returns:
            List of similar texts
        """
        # Get query embedding
        query_embedding = torch.tensor(self.get_embedding(query))
        
        # Get embeddings for all texts in dataset
        text_embeddings = []
        for text in tqdm(self.texts):
            text_embedding = torch.tensor(self.get_embedding(text))
            text_embeddings.append(text_embedding)
        
        # Calculate similarities
        similarities = []
        for text_embedding in text_embeddings:
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                text_embedding.unsqueeze(0)
            )
            similarities.append(similarity.item())
        
        # Get top-k similar texts
        top_indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], 
                           reverse=True)[:top_k]
        
        return [self.texts[i] for i in top_indices] 