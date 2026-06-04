# Product Review Analysis API

A FastAPI microservice that analyzes product reviews using AI/ML:

- **Sentiment Prediction** — classifies reviews as positive/negative using a fine-tuned DistilBERT model (PyTorch).
- **Reason Extraction** — identifies top reasons for negative feedback using a local Ollama LLM (`qwen3.6:35b-a3b`).

## Prerequisites

- **Python** ≥ 3.10
- **Ollama** running locally with the `qwen3.6:35b-a3b` model pulled:
  ```bash
  ollama pull qwen3.6:35b-a3b
  ```
- **Model assets** placed in the `ml_assets/` directory:
  - `ml_assets/distilbert_sentiment.pt` — trained DistilBERT state dict
  - `ml_assets/sentiment_model/` — saved HuggingFace tokenizer files

## Setup & Installation

### Option A: Using `uv` (Recommended)

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

### Option B: Using standard `pip`

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note (Windows):** Replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.

- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Exposing the API with ngrok

To make the API accessible over the internet (e.g., for a frontend deployed elsewhere or for team members to test), you can use [ngrok](https://ngrok.com/) to create a public tunnel to your local server.

### 1. Install ngrok

```bash
# macOS (Homebrew)
brew install ngrok

# Or download from https://ngrok.com/download
```

### 2. Authenticate (one-time setup)

Sign up at [ngrok.com](https://ngrok.com/) and copy your auth token, then run:

```bash
ngrok config add-authtoken <YOUR_AUTH_TOKEN>
```

### 3. Start the tunnel

With the FastAPI server already running on port **8000**, open a **new terminal** and run:

```bash
ngrok http 8000
```

ngrok will display a public URL like:

```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

You can now access the API at `https://abc123.ngrok-free.app/analyze` from anywhere.

> **Tip:** The free tier generates a random URL each time. For a stable URL, use a paid ngrok plan or set a custom domain.

## API Usage

### `POST /analyze`

Send a JSON payload of reviews:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      { "id": 0, "text": "Great phone, love the camera." },
      { "id": 1, "text": "The battery dies too fast." },
      { "id": 2, "text": "Box arrived crushed and charger was missing." }
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
    { "reason": "Battery dies too fast", "count": 1 }
  ],
  "shipping_reasons": [
    { "reason": "Box arrived crushed", "count": 1 },
    { "reason": "Missing charging cable", "count": 1 }
  ],
  "reviews": [
    { "id": 0, "text": "Great phone, love the camera.", "label": "positive" },
    { "id": 1, "text": "The battery dies too fast.", "label": "negative" },
    { "id": 2, "text": "Box arrived crushed and charger was missing.", "label": "negative" }
  ]
}
```

## Testing with Dummy Data

A test script and 100 realistic smartphone reviews are included for end-to-end testing.

### Install test dependency

```bash
pip install requests
```

### Run against local server

```bash
python test_api.py
```

### Run against ngrok URL

```bash
python test_api.py --url https://abc123.ngrok-free.app
```

The script will:
1. ✅ Perform a health check
2. 📄 Load 100 dummy reviews from `test_data/dummy_reviews.json`
3. 🚀 Send them to `POST /analyze`
4. 📊 Print a formatted summary (counts, reasons, labeled reviews)
5. 💾 Save the full JSON response to `test_data/last_response.json`

## Project Structure

```
├── main.py                 # FastAPI app, CORS, and endpoint definitions
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── test_api.py             # Test script for the /analyze endpoint
├── models/
│   └── schemas.py          # Pydantic models for request/response validation
├── services/
│   ├── sentiment.py        # DistilBERT model loading and inference
│   └── extraction.py       # Ollama LLM reason extraction
├── test_data/
│   └── dummy_reviews.json  # 100 realistic smartphone reviews for testing
└── ml_assets/
    ├── distilbert_sentiment.pt  # Trained model weights
    └── sentiment_model/         # Saved tokenizer
```
