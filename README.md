# Neural-Machine-Translation-Transformer

## Project Description
This repository contains a Python-based Jupyter notebook implementing a sequence-to-sequence (seq2seq) Transformer model for Neural Machine Translation (NMT) from French to English. The project, implemented in PyTorch, processes a parallel corpus of French-English sentence pairs, performing preprocessing steps such as tokenization using Hugging Face's `transformers` library. The model is trained on a dataset with splits for training, validation, and testing, and evaluated using the BLEU score with beam search decoding. The notebook includes data loading, model training, and evaluation, with visualizations of translation results. It is compatible with Google Colab or local environments and includes answers to specific questions about vocabulary size, batch token statistics, model checkpointing, translation quality, and beam search optimization.

## Key Features
- **Preprocessing**: Tokenization of French (source) and English (target) sentences using Hugging Face tokenizers.
- **Model**: Transformer-based Encoder-Decoder model with configurable parameters (e.g., hidden size, attention heads, layers).
- **Training**: Training loop with PyTorch, checkpointing the best model based on validation perplexity.
- **Evaluation**: Beam search decoding for translation generation, evaluated with corpus-level BLEU score.
- **Dataset**: Parallel French-English corpus (e.g., from resources.zip), with train, validation, and test splits.
- **Analysis**: Includes EDA, translation quality analysis, and optimization suggestions for beam search efficiency.

## Requirements
- Python 3.12.6
- PyTorch (version compatible with system, e.g., 1.8.0 or later)
- transformers==4.27.0
- datasets==2.10.0
- NLTK==3.8.1
- NumPy, Pandas, Matplotlib, tqdm
- Jupyter Notebook
