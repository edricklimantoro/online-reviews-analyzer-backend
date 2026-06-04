"""
Negative review reason extraction service using a local Ollama LLM.

Sends negative reviews to Ollama (qwen3.6:35b-a3b) and parses structured
JSON output containing product-related and shipping-related reasons.
"""

import json
import logging
import os

import ollama
from ollama import Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "qwen3.6:35b-a3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "")
OLLAMA_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}"

_client = Client(host=OLLAMA_URL)

# Default empty response used when extraction fails
EMPTY_REASONS: dict = {
    "product_reasons": [],
    "shipping_reasons": [],
}


def _build_prompt(negative_reviews: list[str]) -> str:
    """
    Construct the analysis prompt for the LLM.

    The prompt instructs the model to:
    1. Analyze all provided negative reviews.
    2. Extract the top 3 product-related reasons.
    3. Extract the top 3 shipping/packaging/missing-items reasons.
    4. Return ONLY valid JSON in the required schema.
    """
    reviews_block = "\n".join(
        f"  {i + 1}. \"{review}\"" for i, review in enumerate(negative_reviews)
    )

    prompt = f"""You are an expert product review analyst. Below is a list of negative product reviews. Your task is to carefully analyze them and extract the most common reasons customers are unhappy.

NEGATIVE REVIEWS:
{reviews_block}

INSTRUCTIONS:
1. Identify the top 3 reasons the reviews are negative specifically regarding the **Product itself** (e.g., quality, functionality, design, performance).
2. Identify the top 3 reasons the reviews are negative specifically regarding **Shipping, packaging, or missing items** (e.g., damaged box, late delivery, missing accessories).
3. For each reason, provide a short description and the count of how many reviews mention that reason.
4. If there are fewer than 3 reasons in a category, include only the ones you found.
5. If no reasons exist for a category, return an empty array for that category.

OUTPUT FORMAT:
You must respond with ONLY valid JSON. Do not include any explanation, markdown, or text outside the JSON object. Use this exact schema:

{{
  "product_reasons": [
    {{"reason": "short description of the product issue", "count": <number_of_reviews>}},
    {{"reason": "short description of the product issue", "count": <number_of_reviews>}},
    {{"reason": "short description of the product issue", "count": <number_of_reviews>}}
  ],
  "shipping_reasons": [
    {{"reason": "short description of the shipping issue", "count": <number_of_reviews>}},
    {{"reason": "short description of the shipping issue", "count": <number_of_reviews>}},
    {{"reason": "short description of the shipping issue", "count": <number_of_reviews>}}
  ]
}}"""

    return prompt


def _validate_reasons(data: dict) -> dict:
    """
    Validate and normalize the parsed JSON response from the LLM.

    Ensures the response has the expected keys and structure.
    Truncates each category to at most 3 items.
    """
    result = {
        "product_reasons": [],
        "shipping_reasons": [],
    }

    for key in ("product_reasons", "shipping_reasons"):
        items = data.get(key, [])
        if not isinstance(items, list):
            logger.warning("Expected list for '%s', got %s. Skipping.", key, type(items).__name__)
            continue

        validated = []
        for item in items[:3]:  # Take at most 3
            if isinstance(item, dict) and "reason" in item and "count" in item:
                validated.append({
                    "reason": str(item["reason"]),
                    "count": int(item["count"]) if isinstance(item["count"], (int, float)) else 0,
                })
            else:
                logger.warning("Skipping malformed reason item: %s", item)

        result[key] = validated

    return result


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_negative_reasons(negative_reviews: list[str]) -> dict:
    """
    Extract top reasons for negative sentiment from a list of negative reviews.

    Uses a local Ollama instance with the qwen3.6:35b-a3b model to analyze
    the reviews and produce structured JSON output.

    Args:
        negative_reviews: List of review texts already classified as negative.

    Returns:
        dict with keys "product_reasons" and "shipping_reasons", each
        containing a list of {"reason": str, "count": int} dicts.
        Returns empty lists on failure.
    """
    if not negative_reviews:
        logger.info("No negative reviews to analyze. Returning empty reasons.")
        return EMPTY_REASONS.copy()

    prompt = _build_prompt(negative_reviews)

    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction assistant. "
                        "You must respond with ONLY valid JSON, no explanation or additional text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.1,  # Low temperature for deterministic output
            },
        )

        raw_content = response["message"]["content"].strip()
        logger.debug("Ollama raw response: %s", raw_content)

        # Parse the JSON response
        parsed = json.loads(raw_content)
        result = _validate_reasons(parsed)

        logger.info(
            "Extraction complete: %d product reasons, %d shipping reasons.",
            len(result["product_reasons"]),
            len(result["shipping_reasons"]),
        )
        return result

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Ollama response as JSON: %s", e)
        return EMPTY_REASONS.copy()

    except ollama.ResponseError as e:
        logger.error("Ollama API error: %s", e)
        return EMPTY_REASONS.copy()

    except Exception as e:
        logger.error(
            "Unexpected error communicating with Ollama: %s: %s",
            type(e).__name__,
            e,
        )
        return EMPTY_REASONS.copy()
