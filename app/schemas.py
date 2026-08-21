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


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]