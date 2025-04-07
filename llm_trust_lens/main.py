from llm_trust_lens.huggingface_embedding_model import HuggingFaceEmbeddingModel

if __name__ == "__main__":
    huggingface_embedding_model = HuggingFaceEmbeddingModel()
    print(huggingface_embedding_model.retrieve_similar("This movie was scary and I didn't like it"))
