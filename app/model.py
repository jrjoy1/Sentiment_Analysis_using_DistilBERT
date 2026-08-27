import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ==============================
# MODEL PATHS
# ==============================

DISTILBERT_PATH = "mdjrjoy/distilbert-fine-tuned"

XLM_ROBERTA_PATH = "mdjrjoy/xlm-roberta-based"


# ==============================
# LOAD DISTILBERT
# ==============================

distilbert_tokenizer = AutoTokenizer.from_pretrained(
    DISTILBERT_PATH
)

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_PATH
)

distilbert_model.to("cpu")
distilbert_model.eval()


# ==============================
# LOAD XLM-ROBERTA
# ==============================

xlm_roberta_tokenizer = AutoTokenizer.from_pretrained(
    XLM_ROBERTA_PATH
)

xlm_roberta_model = AutoModelForSequenceClassification.from_pretrained(
    XLM_ROBERTA_PATH
)

xlm_roberta_model.to("cpu")
xlm_roberta_model.eval()


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_sentiment(
    text: str,
    model_name: str = "distilbert"
):

    # --------------------------
    # Select model
    # --------------------------

    if model_name == "distilbert":

        tokenizer = distilbert_tokenizer
        model = distilbert_model

    elif model_name == "xlm-roberta":

        tokenizer = xlm_roberta_tokenizer
        model = xlm_roberta_model

    else:

        raise ValueError(
            "Invalid model. Choose 'distilbert' or 'xlm-roberta'."
        )


    # --------------------------
    # Tokenization
    # --------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Remove token_type_ids if a tokenizer provides them.
    # This prevents compatibility problems with models
    # such as DistilBERT.
    inputs.pop("token_type_ids", None)


    # --------------------------
    # Prediction
    # --------------------------

    with torch.no_grad():

        outputs = model(**inputs)


    # --------------------------
    # Probabilities
    # --------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]


    # --------------------------
    # Predicted class
    # --------------------------

    prediction = torch.argmax(
        probabilities
    ).item()


    sentiment = model.config.id2label[prediction].upper()

    confidence = probabilities[prediction].item()


    # --------------------------
    # All probabilities
    # --------------------------

    all_probabilities = {
        model.config.id2label[i]: round(
            probabilities[i].item(),
            4
        )
        for i in range(len(probabilities))
    }


    # --------------------------
    # Return result
    # --------------------------

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": all_probabilities,
        "model": model_name,
    }