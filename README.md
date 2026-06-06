# Product Review Analysis API

A FastAPI microservice that analyzes product reviews using AI/ML:

- **Sentiment Prediction** — classifies reviews as positive/negative with **confidence scores** using a fine-tuned DistilBERT model (PyTorch).
- **Reason Extraction** — identifies grouped (umbrella) reasons for negative feedback with **severity classification** via a local Ollama LLM (`qwen3.6:35b-a3b`).

## Prerequisites

- **Python** ≥ 3.10
- **Ollama** running with the `qwen3.6:35b-a3b` model pulled:
  ```bash
  ollama pull qwen3.6:35b-a3b
  ```
- **Model assets** placed in the `ml_assets/` directory:
  - `ml_assets/distilbert_sentiment.pt` — trained DistilBERT state dict
  - `ml_assets/sentiment_model/` — saved HuggingFace tokenizer files

## Setup & Installation

### Using standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (Windows):** Replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

## Environment Variables

The API reads configuration from environment variables. Copy the provided `.env` file and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | FastAPI server bind host |
| `API_PORT` | `8000` | FastAPI server port |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |

To load the `.env` file before starting the server:

```bash
export $(grep -v '^#' .env | xargs)
uvicorn main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.

- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## API Usage

### `POST /analyze`

Send a JSON payload of reviews:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      { "id": 1, "text": "Great phone, love the camera." },
      { "id": 2, "text": "The battery dies too fast." },
      { "id": 3, "text": "Box arrived crushed and charger was missing." }
    ]
  }'
```

### Example Response

```json
{
  "total_reviews": 3,
  "positive_count": 1,
  "negative_count": 2,
  "product_reasons": [
    {
      "reason": "Battery Life & Thermal Issues",
      "count": 1,
      "severity": "moderate",
      "severity_score": 4,
      "severity_explanation": "Battery draining quickly impacts daily usability and drives returns.",
      "review_ids": [2]
    }
  ],
  "shipping_reasons": [
    {
      "reason": "Shipping Damage & Missing Accessories",
      "count": 1,
      "severity": "moderate",
      "severity_score": 3,
      "severity_explanation": "Damaged packaging and missing items create customer frustration and support overhead.",
      "review_ids": [3]
    }
  ],
  "reviews": [
    { "id": 1, "text": "Great phone, love the camera.", "label": "positive", "confidence": 0.9987, "confidence_level": "high" },
    { "id": 2, "text": "The battery dies too fast.", "label": "negative", "confidence": 0.9992, "confidence_level": "high" },
    { "id": 3, "text": "Box arrived crushed and charger was missing.", "label": "negative", "confidence": 0.8912, "confidence_level": "medium" }
  ]
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `total_reviews` | Number of reviews analyzed |
| `positive_count` / `negative_count` | Count per sentiment |
| `product_reasons` | Grouped negative reasons about the **product** itself |
| `shipping_reasons` | Grouped negative reasons about **shipping/packaging** |
| `reviews` | Individual reviews with sentiment labels |

#### Reason Object

| Field | Description |
|-------|-------------|
| `reason` | Descriptive umbrella reason name |
| `count` | Number of reviews matching this reason |
| `severity` | `"critical"` \| `"moderate"` \| `"minor"` |
| `severity_score` | 1 (minor) to 5 (critical) |
| `severity_explanation` | Explanation of the severity — designed for tooltip/hover display |
| `review_ids` | IDs of reviews that belong to this reason — enables click-to-filter on the dashboard |

#### Review Object

| Field | Description |
|-------|-------------|
| `id` | Review ID from the request |
| `text` | Original review text |
| `label` | `"positive"` or `"negative"` |
| `confidence` | Softmax probability of the predicted class (0.0 – 1.0) |
| `confidence_level` | `"high"` (≥0.90) \| `"medium"` (0.70–0.89) \| `"low"` (<0.70) |

### Validation Errors (422)

Invalid requests return a 422 error with a descriptive message:

```json
{
  "detail": "body -> reviews -> 0 -> text: Review text must not be empty or whitespace only"
}
```

Triggered by: invalid JSON, empty body, missing fields, empty/whitespace-only text, wrong types, empty reviews list.

## Testing

### Happy-path test with dummy data

100 realistic smartphone reviews are included for end-to-end testing:

```bash
python test_api.py
```

Runs against `http://127.0.0.1:8000` by default. For a remote server:

```bash
python test_api.py --url https://your-ngrok-url.ngrok-free.app
```

The script prints a formatted summary and saves the full response to `test_data/last_response.json`.

### Validation error test

12 different malformed request scenarios are tested:

```bash
python test_validation.py
```

All should return 422. Run with `--url` for a remote server.

## Project Structure

```
├── main.py                 # FastAPI app, CORS, exception handlers, and endpoint
├── requirements.txt        # Project dependencies
├── .env                    # Environment configuration (hosts, ports)
├── .gitignore
├── README.md
├── test_api.py             # Happy-path test script
├── test_validation.py      # Validation-error test script
├── models/
│   └── schemas.py          # Pydantic models for request/response validation
├── services/
│   ├── sentiment.py        # DistilBERT model loading and inference with confidence
│   └── extraction.py       # Ollama LLM reason extraction with severity and grouping
├── test_data/
│   └── dummy_reviews.json  # 100 realistic smartphone reviews for testing
└── ml_assets/
    ├── distilbert_sentiment.pt  # Trained model weights
    └── sentiment_model/         # Saved tokenizer
```
