# LLMs

A collection of Jupyter notebooks exploring Large Language Models (LLMs) and NLP tasks using Google Gemini, HuggingFace Transformers, BERT, and XLNet.

## Notebooks

### 1. Gemini Models
Demonstrates Google Gemini API integration including:
- Text generation with configurable temperature and token limits
- Text summarization with few-shot learning
- Poetic chatbot using system instructions and chat history
- Manual Retrieval-Augmented Generation (RAG) with FAISS vector stores over web content

### 2. HuggingFace Transformers
An introduction to the HuggingFace Transformers library covering:
- Sentiment analysis and Named Entity Recognition (NER) pipelines
- Zero-shot text classification
- Tokenization with different tokenizers (BERT, XLNet, DistilBERT)
- Model loading, saving, and PyTorch inference

### 3. Question Answering with BERT
Explores extractive question answering using transformer models:
- Context-based QA with BERT fine-tuned on SQuAD
- Comparison of BERT, RoBERTa, and DistilBERT for QA tasks
- Embeddings and token classification

### 4. Text Classification with XLNet
Fine-tunes XLNet for emotion classification on Twitter data:
- Text preprocessing and cleaning
- Fine-tuning XLNetForSequenceClassification with HuggingFace Trainer
- Evaluation and pipeline-based inference

## Requirements

- Python 3.x
- [google-generativeai](https://pypi.org/project/google-generativeai/)
- [transformers](https://pypi.org/project/transformers/)
- [torch](https://pypi.org/project/torch/)
- [langchain](https://pypi.org/project/langchain/)
- [datasets](https://pypi.org/project/datasets/)
- [evaluate](https://pypi.org/project/evaluate/)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [pandas](https://pypi.org/project/pandas/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [seaborn](https://pypi.org/project/seaborn/)
- [clean-text](https://pypi.org/project/clean-text/)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedtalaat14/LLMs.git
   cd LLMs
   ```
2. Install dependencies:
   ```bash
   pip install google-generativeai transformers torch langchain datasets evaluate faiss-cpu scikit-learn pandas matplotlib seaborn clean-text
   ```
3. Add your Gemini API key in `config.py`:
   ```python
   gemini_key = "YOUR_API_KEY"
   ```
4. Open the notebooks with Jupyter:
   ```bash
   jupyter notebook
   ```