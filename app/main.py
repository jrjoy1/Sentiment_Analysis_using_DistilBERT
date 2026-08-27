from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_sentiment
from app.schemas import SentimentRequest, SentimentResponse


# ==============================
# FASTAPI APP
# ==============================

app = FastAPI(
    title="Bangla & English Sentiment Analysis API",
    description=(
        "Customer review sentiment classification using "
        "DistilBERT and XLM-RoBERTa."
    ),
    version="2.0.0",
)


# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# HOME
# ==============================

@app.get("/")
def home():

    return {
        "message": "Sentiment Analysis API is running locally!",
        "models": [
            "distilbert",
            "xlm-roberta"
        ]
    }


# ==============================
# PREDICT
# ==============================

@app.post(
    "/predict",
    response_model=SentimentResponse
)
def predict(request: SentimentRequest):

    try:

        result = predict_sentiment(
            request.text,
            request.model
        )

        return {
            "text": request.text,
            **result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )