"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator


class ReviewItem(BaseModel):
    """A single product review in the incoming request."""

    id: int = Field(..., description="Unique identifier for the review")
    text: str = Field(..., min_length=1, description="The review text content")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Review text must not be empty or whitespace only")
        return stripped


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    reviews: list[ReviewItem] = Field(
        ..., description="List of product reviews to analyze", min_length=1
    )


class LabeledReview(BaseModel):
    """A review annotated with its predicted sentiment label."""

    id: int
    text: str
    label: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")


class ReasonItem(BaseModel):
    """A single extracted reason with its occurrence count."""

    reason: str = Field(..., description="Description of the reason")
    count: int = Field(..., ge=0, description="Number of reviews mentioning this reason")


class AnalyzeResponse(BaseModel):
    """Full response from the /analyze endpoint."""

    total_reviews: int = Field(..., ge=0)
    positive_count: int = Field(..., ge=0)
    negative_count: int = Field(..., ge=0)
    product_reasons: list[ReasonItem] = Field(
        default_factory=list,
        description="Top reasons for negative reviews related to the product",
    )
    shipping_reasons: list[ReasonItem] = Field(
        default_factory=list,
        description="Top reasons for negative reviews related to shipping/packaging",
    )
    reviews: list[LabeledReview] = Field(
        ..., description="All reviews with their predicted sentiment labels"
    )
