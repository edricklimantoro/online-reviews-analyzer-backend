"""
Negative review reason extraction service using a local Ollama LLM.

Sends negative reviews to Ollama (qwen3.6:35b-a3b) and parses structured
JSON output containing product-related and shipping-related reasons with
severity classification and linked review IDs.
"""

import json
import logging
import os

import ollama
from ollama import Client

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen3.6:35b-a3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

_client = Client(host=OLLAMA_HOST)

EMPTY_REASONS: dict = {
    "product_reasons": [],
    "shipping_reasons": [],
}


def _build_prompt(negative_reviews: list[tuple[int, str]]) -> str:
    """
    Construct the analysis prompt for the LLM.

    Uses multi-step reasoning to:
    1. Infer product type.
    2. Group similar complaints into meaningful umbrella reasons.
    3. Assign severity per reason.
    4. Map review IDs to reasons (only IDs from the provided list).
    5. Self-verify the output before emitting JSON.
    """
    reviews_block = "\n".join(
        f"  {i + 1}. [ID={review_id}] \"{text}\""
        for i, (review_id, text) in enumerate(negative_reviews)
    )

    valid_ids = [str(rid) for rid, _ in negative_reviews]

    prompt = f"""You are an expert product review analyst. Below is a list of negative product reviews, each identified by a unique [ID].

NEGATIVE REVIEWS:
{reviews_block}

Follow these steps carefully, reasoning step by step:

**STEP 1 — Product identification**
Infer what type of product these reviews are about (e.g., smartphone, laptop, clothing, furniture).

**STEP 2 — Thematic grouping**
Read every review and identify the KEY COMPLAINT THEMES. Group similar complaints into ONE REASON:
- Merge minor variations of the same issue into a broader but still descriptive category.
- Example: "battery drains fast" + "battery shuts down early" + "overheats while charging" → merge into "Battery life, charging, and overheating issues" (NOT too general like "Battery problems").
- Each reason name should be specific enough that a product manager understands exactly what to investigate, but broad enough to cover 2+ reviews.
- Aim for 3-7 reasons per category. If you have more than 7, you are over-splitting — merge further.
- If only 1 review describes a truly unique and critical issue (safety hazard, product broken), it can be its own reason. Otherwise, merge it.

**STEP 3 — Assign severity**
Score each reason from 1 (minor) to 5 (critical) considering:
- Safety risk, functionality impact, customer frustration, business impact.
- 1-2 → "minor"
- 3-4 → "moderate"
- 5 → "critical"
Write a short severity_explanation referencing the specific customer and business impact.

**STEP 4 — Link reviews**
For each reason, list the [ID]s of reviews that match. Every review must belong to exactly one product reason and/or one shipping reason (if applicable).

**STEP 5 — SELF-VERIFY (CRITICAL)**
Before outputting, verify all of the following. If ANY check fails, correct your response:
1. Every review_id in your output appears in this list: {valid_ids}
2. Do NOT include IDs that are not in this list.
3. Every review in the input is assigned to at least one reason (product or shipping).
4. No review is assigned to more than one reason within the same category.
5. The "count" field must match the length of the "review_ids" array.
6. Double-check: did I merge similar complaints or did I split too finely?

OUTPUT FORMAT:
Respond with ONLY valid JSON. No explanation, no markdown.

{{
  "product_type": "inferred product category",
  "product_reasons": [
    {{
      "reason": "descriptive but grouped issue name",
      "count": <number_of_reviews>,
      "severity": "critical|moderate|minor",
      "severity_score": <1-5>,
      "severity_explanation": "why this severity — references business and customer impact",
      "review_ids": [<list of review IDs>]
    }}
  ],
  "shipping_reasons": [
    {{
      "reason": "descriptive but grouped issue name",
      "count": <number_of_reviews>,
      "severity": "critical|moderate|minor",
      "severity_score": <1-5>,
      "severity_explanation": "why this severity — references business and customer impact",
      "review_ids": [<list of review IDs>]
    }}
  ]
}}"""

    return prompt


def _validate_reasons(data: dict, valid_ids: set[int]) -> dict:
    """
    Validate and normalize the parsed JSON response from the LLM.

    Filters out hallucinated review IDs that aren't in valid_ids.
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
        for item in items:
            if not isinstance(item, dict) or "reason" not in item or "count" not in item:
                logger.warning("Skipping malformed reason item: %s", item)
                continue

            # Filter review_ids to only include valid IDs
            raw_ids = [
                int(i) for i in item.get("review_ids", [])
                if isinstance(i, (int, float))
            ]
            filtered_ids = [rid for rid in raw_ids if rid in valid_ids]
            dropped = set(raw_ids) - set(filtered_ids)
            if dropped:
                logger.warning(
                    "Dropped hallucinated review IDs %s from reason '%s'",
                    dropped, item.get("reason", ""),
                )

            validated.append({
                "reason": str(item.get("reason", "")),
                "count": len(filtered_ids),
                "severity": str(item.get("severity", "moderate")),
                "severity_score": int(item["severity_score"]) if isinstance(item.get("severity_score"), (int, float)) else 3,
                "severity_explanation": str(item.get("severity_explanation", "")),
                "review_ids": filtered_ids,
            })

        result[key] = validated

    return result


def extract_negative_reasons(negative_reviews: list[tuple[int, str]]) -> dict:
    """
    Extract top reasons for negative sentiment from a list of negative reviews.

    Uses Ollama to analyze the reviews and produce structured JSON output
    including severity classification, explanations, and linked review IDs.

    Args:
        negative_reviews: List of (review_id, review_text) tuples for
                          reviews already classified as negative.

    Returns:
        dict with keys "product_reasons" and "shipping_reasons", each
        containing a list of dicts with reason, count, severity,
        severity_score, severity_explanation, and review_ids.
        Returns empty lists on failure.
    """
    if not negative_reviews:
        logger.info("No negative reviews to analyze. Returning empty reasons.")
        return EMPTY_REASONS.copy()

    prompt = _build_prompt(negative_reviews)

    valid_ids = {rid for rid, _ in negative_reviews}

    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction and analysis assistant. "
                        "You must respond with ONLY valid JSON, no explanation or additional text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.1,
            },
        )

        raw_content = response["message"]["content"].strip()
        logger.debug("Ollama raw response: %s", raw_content)

        parsed = json.loads(raw_content)
        result = _validate_reasons(parsed, valid_ids)

        # Log any input IDs not assigned to any reason (only for informational purposes)
        all_assigned = set()
        for key in ("product_reasons", "shipping_reasons"):
            for r in result[key]:
                all_assigned.update(r["review_ids"])
        unassigned = valid_ids - all_assigned
        if unassigned:
            logger.warning("Reviews %s were not assigned to any reason.", sorted(unassigned))

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
