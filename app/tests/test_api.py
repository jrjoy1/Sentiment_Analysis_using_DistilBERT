from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ==============================
# HOME TEST
# ==============================

def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "models" in data

    assert "distilbert" in data["models"]
    assert "xlm-roberta" in data["models"]


# ==============================
# DISTILBERT POSITIVE
# ==============================

def test_distilbert_positive_sentiment():

    response = client.post(
        "/predict",
        json={
            "text": "I really love this product!",
            "model": "distilbert"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "POSITIVE"
    assert data["model"] == "distilbert"

    assert 0 <= data["confidence"] <= 1


# ==============================
# DISTILBERT NEGATIVE
# ==============================

def test_distilbert_negative_sentiment():

    response = client.post(
        "/predict",
        json={
            "text": "This product is terrible and useless.",
            "model": "distilbert"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "NEGATIVE"
    assert data["model"] == "distilbert"

    assert 0 <= data["confidence"] <= 1


# ==============================
# XLM-ROBERTA BANGLA POSITIVE
# ==============================

def test_xlm_roberta_positive_sentiment():

    response = client.post(
        "/predict",
        json={
            "text": "এই পণ্যটি অনেক ভালো",
            "model": "xlm-roberta"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "POSITIVE"
    assert data["model"] == "xlm-roberta"

    assert 0 <= data["confidence"] <= 1


# ==============================
# XLM-ROBERTA BANGLA NEGATIVE
# ==============================

def test_xlm_roberta_negative_sentiment():

    response = client.post(
        "/predict",
        json={
            "text": "এই পণ্যটি একদম খারাপ",
            "model": "xlm-roberta"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "NEGATIVE"
    assert data["model"] == "xlm-roberta"

    assert 0 <= data["confidence"] <= 1


# ==============================
# EMPTY TEXT
# ==============================

def test_empty_text():

    response = client.post(
        "/predict",
        json={
            "text": "",
            "model": "distilbert"
        }
    )

    assert response.status_code == 422


# ==============================
# TEXT TOO LONG
# ==============================

def test_text_too_long():

    response = client.post(
        "/predict",
        json={
            "text": "a" * 1001,
            "model": "distilbert"
        }
    )

    assert response.status_code == 422


# ==============================
# INVALID MODEL
# ==============================

def test_invalid_model():

    response = client.post(
        "/predict",
        json={
            "text": "This product is good.",
            "model": "invalid-model"
        }
    )

    assert response.status_code == 422