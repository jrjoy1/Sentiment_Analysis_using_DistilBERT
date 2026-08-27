from typing import Literal

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text to analyze",
        json_schema_extra={
            "example": "This product is excellent!"
        }
    )

    model: Literal["distilbert", "xlm-roberta"] = Field(
        default="distilbert",
        description="Model to use for sentiment analysis"
    )


class SentimentResponse(BaseModel):

    text: str

    sentiment: str

    confidence: float

    probabilities: dict[str, float]

    model: str