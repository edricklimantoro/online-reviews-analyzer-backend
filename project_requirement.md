
# Project Overview
You are an expert Python engineer and Machine Learning backend developer. Your task is to build a complete FastAPI backend project that serves as an AI/ML microservice for a product review analysis system.

The API will receive a JSON payload of product reviews, predict their sentiment using a custom PyTorch DistilBERT model, and extract specific reasons for negative reviews using a local Ollama LLM (`qwen3.6:35b-a3b`).

## Technical Stack
* **Web Framework:** FastAPI, Uvicorn
* **Machine Learning:** PyTorch (`torch`), HuggingFace Transformers (`transformers`)
* **LLM Integration:** Ollama Python Library (`ollama`) or raw `requests` to local Ollama instance
* **Package Management & Environment:** `uv` (Astral) or standard Python `venv`

## Directory Structure
Please generate the complete code for the following project structure:
```text
/
├── main.py                 # FastAPI application, CORS, and endpoint definitions
├── requirements.txt        # Project dependencies
├── README.md               # Setup and run instructions
├── models/
│   └── schemas.py          # Pydantic models for request/response validation
├── services/
│   ├── sentiment.py        # Logic for loading DistilBERT and running inference
│   └── extraction.py       # Logic for prompting Ollama to extract reasons
└── ml_assets/              
    ├── distilbert_sentiment.pt  # The state_dict (assumed to exist)
    └── sentiment_model/         # The saved tokenizer (assumed to exist)

```

## Environment & Setup (`uv`)

This project will be run locally on the host machine (not in Docker) to ensure direct access to local GPU resources and the local Ollama instance.

* The project must use standard Python virtual environments.
* The dependencies in `requirements.txt` should be optimized for the fast `uv` package manager (or standard `pip` if `uv` is unavailable).
* You must generate a `README.md` file that includes the exact terminal commands required to:
1. Create the virtual environment (e.g., `uv venv`).
2. Activate the virtual environment.
3. Install dependencies using `uv pip install -r requirements.txt` (or standard `pip`).
4. Start the FastAPI server using `uvicorn main:app --reload`.


## 1. PyTorch Sentiment Model Requirements

The sentiment model was fine-tuned and saved using a specific custom class wrapper. To load the weights successfully (`distilbert_sentiment.pt`), **you must use the exact class definition below** in your `services/sentiment.py` file:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased" # Assuming base model name

class DistilBert(nn.Module):
    def __init__(self, pretrained_model_name: str = MODEL_NAME, num_classes: int = 2):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.distilbert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        
        hidden_size = config.hidden_size
        dropout_prob = getattr(config, "classifier_dropout", getattr(config, "hidden_dropout_prob", 0.2))
        
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]  
        pooled_output = hidden_state[:, 0]  
        pooled_output = self.pre_classifier(pooled_output)  
        pooled_output = F.relu(pooled_output)  
        pooled_output = self.dropout(pooled_output)  
        logits = self.classifier(pooled_output)  

        return logits

```

**Instructions for `services/sentiment.py`:**

* Initialize this `DistilBert` class.
* Load the state dictionary: `model.load_state_dict(torch.load("ml_assets/distilbert_sentiment.pt", map_location=device))`
* Set the model to evaluation mode (`model.eval()`).
* Load the tokenizer using: `AutoTokenizer.from_pretrained("ml_assets/sentiment_model")`
* Create a function `predict_sentiment(texts: list[str]) -> list[str]` that processes a list of reviews and returns "positive" or "negative" for each based on the logit outputs (assume index 1 is positive, index 0 is negative, or vice versa, clearly defined).

## 2. LLM Extraction Requirements (Ollama)

* Model to use: **`qwen3.6:35b-a3b`**.
* In `services/extraction.py`, create a function that takes a list of strictly **negative** reviews.
* Construct a prompt for Ollama that instructs it to analyze the negative reviews and extract:
1. The top 3 reasons the reviews were bad specifically regarding the **Product itself**.
2. The top 3 reasons the reviews were bad specifically regarding the **Shipping, packaging, or missing items**.


* The prompt must strictly instruct Ollama to output ONLY valid JSON in this exact schema:
```json
{
  "product_reasons": [{"reason": "string", "count": 0}],
  "shipping_reasons": [{"reason": "string", "count": 0}]
}

```


## 3. API Endpoint Specification

**Endpoint:** `POST /analyze`

**Input (JSON Payload):**
The endpoint must accept a JSON request body (using Pydantic models in `models/schemas.py`).
```json
{
  "reviews": [
    { "id": 0, "text": "Great phone, love the camera." },
    { "id": 1, "text": "The battery dies too fast." }
  ]
}
```

**Processing Flow:**

1. Validate the incoming JSON payload using Pydantic.
2. Extract the review texts and pass them to predict_sentiment(). Maintain the association between each review's id and its new sentiment label.
3. Calculate the total_reviews, positive_count, and negative_count based on the sentiment predictions.
3. Filter out only the "negative" reviews.
4. Pass the negative reviews to the Ollama extraction service.
5. Format the final output to match the consumer's expected JSON format, ensuring the original id is included in the returned reviews array.

**Output (JSON):**
The downstream consumer expects the negative reasons separated by category. You must output the top 3 product-related reasons and the top 3 shipping-related reasons into their own respective arrays (`product_reasons` and `shipping_reasons`). Do not combine them. The JSON strict format is given in the example below.

```json
{
  "total_reviews": 100,
  "positive_count": 80,
  "negative_count": 20,
  "product_reasons": [
    { "reason": "Battery dies too fast", "count": 12 },
    { "reason": "Screen scratches easily", "count": 4 },
    { "reason": "Overheats during use", "count": 2 }
  ],
  "shipping_reasons": [
    { "reason": "Box arrived crushed", "count": 5 },
    { "reason": "Missing charging cable", "count": 3 },
    { "reason": "Delivery was delayed by a week", "count": 1 }
  ],
  "reviews": [
    { "id": 0, "text": "Great phone, love the camera.", "label": "positive" },
    { "id": 1, "text": "The battery dies too fast.", "label": "negative" }
  ]
}
```

## Deliverables

Please generate the complete code for:

1. `README.md (with uv/venv setup instructions)`
2. `requirements.txt`
3. `main.py`
4. `models/schemas.py`
5. `services/sentiment.py`
6. `services/extraction.py`
Include robust error handling (e.g., if Ollama is unreachable, or if the JSON is malformed).
