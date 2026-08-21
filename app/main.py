from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_sentiment
from app.schemas import SentimentRequest, SentimentResponse


app = FastAPI(
    title="Sentiment Analysis API",
    description="Customer review sentiment classification using DistilBERT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running locally!"
    }


@app.post(
    "/predict",
    response_model=SentimentResponse
)
def predict(request: SentimentRequest):

    try:
        result = predict_sentiment(request.text)

        return {
            "text": request.text,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )