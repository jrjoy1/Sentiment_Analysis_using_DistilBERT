import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_PATH = "mdjrjoy/distilbert-fine-tuned"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to("cpu")
model.eval()


def predict_sentiment(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # DistilBERT does not need token_type_ids
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    sentiment = model.config.id2label[prediction]

    confidence = probabilities[prediction].item()

    all_probabilities = {
        model.config.id2label[i]: round(
            probabilities[i].item(),
            4
        )
        for i in range(len(probabilities))
    }

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": all_probabilities,
    }