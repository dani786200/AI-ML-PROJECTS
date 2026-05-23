# 📰 Task 1: News Topic Classifier Using BERT
**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

## Objective
Fine-tune `bert-base-uncased` to classify news headlines into 4 topic categories using the AG News dataset.

## Methodology / Approach
- **Tokenization:** `BertTokenizer` with max_length=128 and padding
- **Model:** `BertForSequenceClassification` (bert-base-uncased, 110M params)
- **Training:** HuggingFace `Trainer` API, 3 epochs, lr=2e-5, AdamW optimizer, warmup steps=100
- **Evaluation:** Accuracy, weighted F1-Score, per-class F1, confusion matrix
- **Deployment:** Gradio web app for live headline classification

## Dataset
AG News (Hugging Face Datasets) — 4 classes: World, Sports, Business, Sci/Tech

## Key Results / Observations
- **Accuracy:** ~92%+ on test set (full dataset → ~94%)
- **F1-Score (weighted):** ~0.92
- Sports & Business are easiest to classify; World & Sci/Tech occasionally overlap
- 3 epochs of fine-tuning is sufficient for convergence on AG News

## Files
| File | Description |
|---|---|
| `Task1_News_Classifier_BERT.ipynb` | Full solution notebook |
| `bert_news_classifier_final/` | Saved fine-tuned model & tokenizer |
| `ag_news_eda.png` | EDA visualizations |
| `bert_evaluation.png` | Confusion matrix & per-class F1 |

## Setup
```bash
pip install transformers datasets torch gradio scikit-learn
jupyter notebook Task1_News_Classifier_BERT.ipynb
```

## Skills Demonstrated
NLP with Transformers · Transfer learning & fine-tuning · Evaluation metrics for text classification · Lightweight model deployment (Gradio)
