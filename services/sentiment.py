"""
Sentiment prediction service using a fine-tuned DistilBERT model.

The DistilBert class below is the EXACT architecture used during training.
It must remain unchanged to ensure load_state_dict() compatibility with
the saved weights in ml_assets/distilbert_sentiment.pt.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Paths to model assets (relative to project root)
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"
MODEL_WEIGHTS_PATH = Path("ml_assets/distilbert_sentiment.pt")
TOKENIZER_PATH = Path("ml_assets/sentiment_model")

# Label mapping: index 0 -> negative, index 1 -> positive
LABEL_MAP = {0: "negative", 1: "positive"}


# ---------------------------------------------------------------------------
# Model architecture (MUST match the training code exactly)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model loading (called once at application startup)
# ---------------------------------------------------------------------------
def load_model() -> tuple:
    """
    Loads the DistilBERT model and tokenizer from disk.

    Returns:
        tuple: (model, tokenizer) ready for inference.

    Raises:
        FileNotFoundError: If model weights or tokenizer directory is missing.
        RuntimeError: If state_dict loading fails (architecture mismatch).
    """
    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_WEIGHTS_PATH}'. "
            "Please place the trained distilbert_sentiment.pt file in ml_assets/."
        )
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at '{TOKENIZER_PATH}'. "
            "Please place the saved tokenizer files in ml_assets/sentiment_model/."
        )

    logger.info("Loading DistilBERT model on device: %s", device)

    # Instantiate and load weights
    model = DistilBert(pretrained_model_name=MODEL_NAME, num_classes=2)
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

    logger.info("DistilBERT model and tokenizer loaded successfully.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def confidence_level(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


@torch.no_grad()
def predict_sentiment(
    texts: list[str],
    model: nn.Module,
    tokenizer: AutoTokenizer,
) -> list[tuple[str, float]]:
    """
    Predict sentiment labels and confidence scores for a list of review texts.

    Args:
        texts: List of review strings.
        model: Loaded DistilBert model in eval mode.
        tokenizer: The corresponding tokenizer.

    Returns:
        List of (label, confidence) tuples, one per input text.
        Confidence is the softmax probability of the predicted class.
    """
    if not texts:
        return []

    # Tokenize the batch
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Forward pass
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = F.softmax(logits, dim=1)
    confidence, predictions = torch.max(probs, dim=1)

    # Map indices to labels and pair with confidence
    results = [
        (LABEL_MAP[pred], conf)
        for pred, conf in zip(predictions.tolist(), confidence.tolist())
    ]
    return results
