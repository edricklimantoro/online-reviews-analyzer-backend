"""
Test script for the Product Review Analysis API.

Loads dummy review data from test_data/dummy_reviews.json, sends it to the
/analyze endpoint, and prints a formatted summary of the results.

Usage:
    python test_api.py                          # default: http://127.0.0.1:8000
    python test_api.py --url https://abc123.ngrok-free.app   # ngrok URL
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
DEFAULT_URL = f"http://{API_HOST}:{API_PORT}"
DUMMY_DATA_PATH = Path(__file__).parent / "test_data" / "dummy_reviews.json"


def load_dummy_data() -> dict:
    """Load the dummy reviews JSON file."""
    if not DUMMY_DATA_PATH.exists():
        print(f"❌ Dummy data not found at: {DUMMY_DATA_PATH}")
        sys.exit(1)

    with open(DUMMY_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📄 Loaded {len(data['reviews'])} reviews from {DUMMY_DATA_PATH.name}")
    return data


def test_health(base_url: str) -> bool:
    """Check if the API is reachable."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"✅ Health check passed: {resp.json()}")
            return True
        else:
            print(f"⚠️  Health check returned status {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {base_url}. Is the server running?")
        return False


def test_analyze(base_url: str, payload: dict) -> dict | None:
    """Send reviews to the /analyze endpoint and return the response."""
    url = f"{base_url}/analyze"
    print(f"\n🚀 Sending {len(payload['reviews'])} reviews to POST {url} ...")

    start_time = time.time()
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "69420"}, # ngrok header to suppress browser warning (if applicable)
            timeout=120,  # LLM extraction can take a while
        )
        elapsed = time.time() - start_time

        if resp.status_code == 200:
            print(f"✅ Response received in {elapsed:.2f}s")
            return resp.json()
        else:
            print(f"❌ Request failed with status {resp.status_code}")
            print(f"   Detail: {resp.text}")
            return None

    except requests.Timeout:
        print("❌ Request timed out (>120s). The LLM extraction may need more time.")
        return None
    except requests.ConnectionError:
        print(f"❌ Connection failed to {url}")
        return None


def print_results(result: dict):
    """Pretty-print the analysis results."""
    print("\n" + "=" * 60)
    print("📊  ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\n  Total Reviews:    {result['total_reviews']}")
    print(f"  Positive Count:   {result['positive_count']}")
    print(f"  Negative Count:   {result['negative_count']}")

    pos_pct = (result['positive_count'] / result['total_reviews'] * 100) if result['total_reviews'] > 0 else 0
    neg_pct = (result['negative_count'] / result['total_reviews'] * 100) if result['total_reviews'] > 0 else 0
    print(f"  Positive Rate:    {pos_pct:.1f}%")
    print(f"  Negative Rate:    {neg_pct:.1f}%")

    # Product reasons
    print(f"\n{'─' * 60}")
    print("🔧  TOP PRODUCT REASONS (Negative)")
    print(f"{'─' * 60}")
    if result.get("product_reasons"):
        for i, reason in enumerate(result["product_reasons"], 1):
            print(f"  {i}. {reason['reason']} (mentioned {reason['count']}x)")
    else:
        print("  (none extracted)")

    # Shipping reasons
    print(f"\n{'─' * 60}")
    print("📦  TOP SHIPPING REASONS (Negative)")
    print(f"{'─' * 60}")
    if result.get("shipping_reasons"):
        for i, reason in enumerate(result["shipping_reasons"], 1):
            print(f"  {i}. {reason['reason']} (mentioned {reason['count']}x)")
    else:
        print("  (none extracted)")

    # Sample labeled reviews
    print(f"\n{'─' * 60}")
    print("📝  SAMPLE LABELED REVIEWS (first 10)")
    print(f"{'─' * 60}")
    for review in result["reviews"][:10]:
        emoji = "👍" if review["label"] == "positive" else "👎"
        text_preview = review["text"][:70] + ("..." if len(review["text"]) > 70 else "")
        print(f"  {emoji} [{review['id']:>3}] {text_preview}")

    # Summary of all labels
    print(f"\n{'─' * 60}")
    print("📋  ALL REVIEW LABELS")
    print(f"{'─' * 60}")
    for review in result["reviews"]:
        marker = "+" if review["label"] == "positive" else "-"
        print(f"  [{marker}] ID {review['id']:>3}: {review['text'][:80]}")

    print(f"\n{'=' * 60}")
    print("✅  Test complete!")
    print(f"{'=' * 60}\n")


def save_response(result: dict):
    """Save the full response to a JSON file for inspection."""
    output_path = Path(__file__).parent / "test_data" / "last_response.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Full response saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test the Product Review Analysis API")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the API (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"🎯 Target API: {base_url}")
    print(f"{'─' * 60}")

    # Step 1: Health check
    if not test_health(base_url):
        print("\n💡 Make sure the server is running:")
        print("   uvicorn main:app --reload")
        print(f"\n💡 If using ngrok, pass the URL:")
        print("   python test_api.py --url https://your-id.ngrok-free.app")
        sys.exit(1)

    # Step 2: Load dummy data
    payload = load_dummy_data()

    # Step 3: Send to /analyze
    result = test_analyze(base_url, payload)
    if result is None:
        sys.exit(1)

    # Step 4: Print results
    print_results(result)

    # Step 5: Save full response
    save_response(result)


if __name__ == "__main__":
    main()
