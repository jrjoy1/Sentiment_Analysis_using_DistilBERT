from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_positive_sentiment():
    response = client.post(
        "/predict",
        json={"text": "I really love this product!"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "positive"
    assert 0 <= data["confidence"] <= 1


def test_negative_sentiment():
    response = client.post(
        "/predict",
        json={"text": "This product is terrible and useless."}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "negative"
    assert 0 <= data["confidence"] <= 1


def test_neutral_sentiment():
    response = client.post(
        "/predict",
        json={"text": "The product is okay, nothing special."}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "neutral"
    assert 0 <= data["confidence"] <= 1

def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_text_too_long():
    response = client.post(
        "/predict",
        json={"text": "a" * 1001}
    )

    assert response.status_code == 422