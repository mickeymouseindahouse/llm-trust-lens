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
