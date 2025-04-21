"""
Example script demonstrating the use of OllamaEmbeddingModel.
"""

import sys
import os

import numpy as np

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_trust_lens.ollama_embedding_model import OllamaEmbeddingModel

def main():
    # Initialize the Ollama embedding model
    # Make sure you have Ollama running with the llama3.2 model
    # e.g., by running "ollama run llama3.2" in a separate terminal
    model = OllamaEmbeddingModel(
        model_name="llama3.2",
        api_url="http://localhost:11434",
        dataset_name="uclanlp/wino_bias",
        dataset_config="type1_anti",
        text_column="tokens"
    )

    biased_model = OllamaEmbeddingModel(
        model_name="llama3.2",
        api_url="http://localhost:11434",
        dataset_name="uclanlp/wino_bias",
        dataset_config="type1_pro",
        text_column="tokens"
    )
    
    sample_text = "This is a test sentence to get embeddings from Ollama."
    embedding = model.get_embedding(sample_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First few values: {embedding[:5]}")
    
    query = "The doctor examined the patient."
    similar_texts = model.retrieve_similar(query, top_k=3)
    print("\nQuery:", query)
    print("Similar unbiased texts (anti):")
    for i, text in enumerate(similar_texts, 1):
        print(f"{i}. {text}")

    similar_biased_texts = biased_model.retrieve_similar(query, top_k=3)
    print("\nQuery:", query)
    print("Similar biased texts (pro):")
    for i, text in enumerate(similar_biased_texts, 1):
        print(f"{i}. {text}")

if __name__ == "__main__":
    sentences = ['The physician told the baker that he had cancer',
                 'The physician told the baker that she had cancer']
    model = OllamaEmbeddingModel(
        model_name="llama3.2",
        api_url="http://localhost:11434",
        dataset_name="uclanlp/wino_bias",
        dataset_config="type1_anti",
        text_column="tokens"
    )
    emb_biased = model.get_embedding(sentences[0])
    emb_unbiased = model.get_embedding(sentences[1])
    diff = np.array(emb_biased) - np.array(emb_unbiased)
    print(np.max(diff))
    print(np.min(diff))
    print(np.mean(diff))
    for i in range(np.shape(diff)[0]):
        if np.abs(diff[i]) > 0.4:
            print(f"dim {i} is important")

    # main()
