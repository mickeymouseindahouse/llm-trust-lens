# LLM TrustLens

A framework for evaluating trustworthiness in Large Language Models (LLMs) at the embedding level.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-trust-lens.git
cd llm-trust-lens
```

2. Create and activate a conda environment:
```bash
conda create -n llm-trust python=3.9
conda activate llm-trust
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Using with Ollama

LLM TrustLens supports using Ollama for embedding generation and similarity retrieval:

1. Install Ollama from [https://ollama.com/](https://ollama.com/)

2. Run your desired model in a separate terminal:
```bash
ollama run llama3.2
```

3. Use the OllamaEmbeddingModel in your code:
```python
from llm_trust_lens import OllamaEmbeddingModel

model = OllamaEmbeddingModel(
    model_name="llama3.2",
    api_url="http://localhost:11434",  # Default Ollama API URL
    dataset_name="uclanlp/wino_bias",
    dataset_config="type1_anti",
    text_column="tokens"
)
```

Note: The OllamaEmbeddingModel sends requests to your local Ollama instance running on http://localhost:11434 by default. Make sure Ollama is running before using this model.

## Features

### Bias Evaluation
- Word Embedding Association Test (WEAT)
- Demographic bias detection
- Sentiment bias analysis

### Retrieval Evaluation
- Precision, recall, and F1 scoring
- Top-k retrieval effectiveness
- Semantic similarity assessment

### Hallucination Detection
- Claim-evidence matching
- Semantic similarity thresholding
- Support verification

## License

MIT License
