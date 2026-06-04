"""
FastAPI application for product review sentiment analysis.

This microservice:
1. Receives a JSON payload of product reviews.
2. Predicts sentiment (positive/negative) using a fine-tuned DistilBERT model.
3. Extracts top reasons for negative reviews using a local Ollama LLM.
4. Returns a structured JSON response with counts, reasons, and labeled reviews.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    LabeledReview,
    ReasonItem,
)
from services.extraction import extract_negative_reasons
from services.sentiment import load_model, predict_sentiment

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model at startup so it's ready for inference requests."""
    logger.info("Starting up — loading sentiment model...")
    try:
        model, tokenizer = load_model()
        app.state.model = model
        app.state.tokenizer = tokenizer
        logger.info("Sentiment model loaded successfully.")
    except FileNotFoundError as e:
        logger.error("Model assets missing: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to load sentiment model: %s", e)
        raise

    yield  # Application is running

    logger.info("Shutting down — cleaning up resources.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Product Review Analysis API",
    description=(
        "AI/ML microservice that predicts review sentiment and extracts "
        "reasons for negative feedback using DistilBERT and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: object, exc: RequestValidationError
):
    errors = exc.errors()
    messages = []
    for error in errors:
        loc = " -> ".join(str(l) for l in error["loc"])
        msg = error["msg"]
        ctx = error.get("ctx")
        if ctx and "error" in ctx:
            messages.append(f"{loc}: {ctx['error']}")
        else:
            messages.append(f"{loc}: {msg}")

    logger.warning("Validation error: %s", "; ".join(messages))
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(messages)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_reviews(request: AnalyzeRequest):
    """
    Analyze a list of product reviews.

    1. Predict sentiment for each review using the DistilBERT model.
    2. Count positive and negative reviews.
    3. Extract top reasons for negative reviews via the Ollama LLM.
    4. Return the full structured response.
    """
    reviews = request.reviews
    texts = [r.text for r in reviews]

    # ── Step 1: Sentiment prediction ──────────────────────────────────────
    try:
        labels = predict_sentiment(
            texts=texts,
            model=app.state.model,
            tokenizer=app.state.tokenizer,
        )
    except Exception as e:
        logger.error("Sentiment prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment prediction error: {e}",
        )

    # ── Step 2: Build labeled reviews and count sentiments ────────────────
    labeled_reviews = [
        LabeledReview(id=review.id, text=review.text, label=label)
        for review, label in zip(reviews, labels)
    ]

    total_reviews = len(labeled_reviews)
    positive_count = sum(1 for lr in labeled_reviews if lr.label == "positive")
    negative_count = total_reviews - positive_count

    # ── Step 3: Extract reasons from negative reviews ─────────────────────
    negative_texts = [lr.text for lr in labeled_reviews if lr.label == "negative"]

    product_reasons = []
    shipping_reasons = []

    if negative_texts:
        try:
            reasons = extract_negative_reasons(negative_texts)
            product_reasons = [
                ReasonItem(**item) for item in reasons.get("product_reasons", [])
            ]
            shipping_reasons = [
                ReasonItem(**item) for item in reasons.get("shipping_reasons", [])
            ]
        except Exception as e:
            logger.error("Reason extraction failed: %s", e)
            # Non-fatal: return response without reasons rather than failing
            product_reasons = []
            shipping_reasons = []

    # ── Step 4: Assemble response ─────────────────────────────────────────
    response = AnalyzeResponse(
        total_reviews=total_reviews,
        positive_count=positive_count,
        negative_count=negative_count,
        product_reasons=product_reasons,
        shipping_reasons=shipping_reasons,
        reviews=labeled_reviews,
    )

    logger.info(
        "Analysis complete: %d total, %d positive, %d negative.",
        total_reviews,
        positive_count,
        negative_count,
    )

    return response


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
