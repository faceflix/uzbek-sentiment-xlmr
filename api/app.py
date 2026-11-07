import re

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "yusuf0902/uz-sentiment-xlmr"
LABELS = {0: "salbiy", 1: "ijobiy"}  # 0 = negative, 1 = positive

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()


def norm(s: str) -> str:
    # training'dagi normalizatsiya bilan bir xil bo'lishi shart
    s = s.lower()
    s = re.sub(r"[‘’ʻʼ`´]", "'", s)
    return re.sub(r"\s+", " ", s).strip()


app = FastAPI(title="Uzbek Sentiment API")


class Item(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Item):
    x = tok(norm(item.text), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**x).logits, dim=-1)[0]
    i = int(probs.argmax())
    return {"label": i, "sentiment": LABELS[i], "confidence": round(float(probs[i]), 4)}
