# Uzbek Sentiment Analysis: XLM-R vs. TF-IDF baseline

Binary sentiment classification of informal Uzbek (Latin script) product reviews. I fine-tuned `xlm-roberta-base` with PyTorch and Hugging Face, compared it with a char n-gram TF-IDF + Logistic Regression baseline, analysed the errors, and served the model through a small FastAPI app.

**Model on Hugging Face:** https://huggingface.co/yusuf0902/uz-sentiment-xlmr

**Stack:** Python, PyTorch, Hugging Face Transformers, scikit-learn, FastAPI

## Data

[`behbudiy/uzbek-sentiment-analysis`](https://huggingface.co/datasets/behbudiy/uzbek-sentiment-analysis): 10k short reviews, labels `0` (negative) / `1` (positive). Only a train split exists, so I made my own split.

Preprocessing:
- lowercasing, apostrophe variants unified, whitespace cleanup (no spelling correction; the data is deliberately left noisy);
- duplicate texts removed, and texts that appear with conflicting labels removed, to avoid leakage between splits (about 8.3k rows remain);
- stratified split 80 / 10 / 10, seed 42. Test set: 827 reviews (507 negative, 320 positive).

## Models

1. **Baseline:** char n-gram (2-5) TF-IDF + Logistic Regression. Character n-grams were chosen because the text has many spelling variants.
2. **XLM-RoBERTa base**, fine-tuned for 3 epochs (batch size 32, lr 2e-5, fp16, `max_length=128`), best checkpoint by validation macro-F1.

Training time for the same XLM-R run: **~9m45s on a Colab GPU vs. ~4h12m on CPU** (about 26x faster).

## Results (test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF char n-gram + Logistic Regression | 0.904 | 0.897 |
| **XLM-R fine-tune** | **0.912** | **0.907** |

Per class (XLM-R): negative P 0.927 / R 0.929 / F1 0.928 (n=507); positive P 0.887 / R 0.884 / F1 0.886 (n=320).

**The improvement over the baseline is not statistically significant:** bootstrap 95% CI for the macro-F1 difference is [-0.014, 0.035] (1000 resamples of the test set). On this small, noisy dataset a simple char n-gram model is a strong competitor.

## Error analysis

Patterns found by reading misclassified test reviews (from the first training run):
- **Negated negatives** such as "yomon emas" / "yomonmas" ("not bad"), labeled positive but predicted negative.
- **Mixed sentiment** ("yoqdi, lekin ..."), where the dataset labels themselves are inconsistent.
- **Slang and code-switching** (Russian words, a few English reviews) and **spelling variants** ("yaxwi").
- **Ambiguous labels**, e.g. very short reviews like "nozik juda".

Many errors therefore reflect label noise and mixed sentiment rather than only model weakness, which limits the achievable score on this dataset.

## Use the model

```python
from transformers import pipeline

clf = pipeline("text-classification", model="yusuf0902/uz-sentiment-xlmr")
print(clf("yaxshi mahsulot ekan, tavsiya qilaman"))
```

(`LABEL_1` = positive, `LABEL_0` = negative.)

## Run the API locally

```bash
cd api
pip install -r requirements.txt
pip install torch
python -m uvicorn app:app --port 8000
```

The first start downloads the model (about 1.1 GB) from the Hugging Face Hub. Then open `http://localhost:8000/docs` or send:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "yaxshi mahsulot ekan"}'
```

Response format: `{"label": 1, "sentiment": "ijobiy", "confidence": <float>}`.

## Limitations

- Trained on product reviews only; other domains (news, social media) are untested.
- Single training run with fixed hyperparameters, one seed; results can shift slightly between runs.
- Latin-script Uzbek only.
- Test set is small (827), so small differences between models are within noise.

## Files

- `uz_sentiment.ipynb`: data cleaning, baseline, XLM-R training, evaluation, error analysis
- `api/app.py`, `api/requirements.txt`: FastAPI inference service
