"""
Test validation error cases for the /analyze endpoint.

Sends various malformed requests and asserts 422 responses.

Usage:
    python test_validation.py                          # default: http://127.0.0.1:8000
    python test_validation.py --url https://abc123.ngrok-free.app
"""

import argparse
import json
import os
import sys

import requests

API_HOST = os.getenv("API_HOST", "")
API_PORT = os.getenv("API_PORT", "")
DEFAULT_URL = f"http://{API_HOST}:{API_PORT}"

PASS = 0
FAIL = 0


def test(name: str, base_url: str, raw_body: str | None, content_type: str = "application/json"):
    global PASS, FAIL
    url = f"{base_url}/analyze"
    headers = {"Content-Type": content_type}

    try:
        if raw_body is not None:
            resp = requests.post(url, data=raw_body.encode(), headers=headers, timeout=10)
        else:
            resp = requests.post(url, headers=headers, timeout=10)
    except requests.ConnectionError:
        print(f"  FAIL - cannot connect to {base_url}")
        FAIL += 1
        return

    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text

    if resp.status_code == 422:
        print(f"  PASS (422) - detail: {detail}")
        PASS += 1
    else:
        print(f"  FAIL (got {resp.status_code}, expected 422) - detail: {detail}")
        FAIL += 1


def main():
    parser = argparse.ArgumentParser(description="Test validation error cases")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL (default: {DEFAULT_URL})")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code != 200:
            print("Server not healthy. Make sure uvicorn is running.")
            sys.exit(1)
    except requests.ConnectionError:
        print("Cannot connect. Make sure uvicorn is running.")
        sys.exit(1)

    print(f"Testing {base_url}/analyze\n")

    # 1. Invalid JSON (truncated)
    test("invalid JSON - truncated", base_url, '{"reviews": [{"id": 1, "text"')
    # 2. Invalid JSON - garbage
    test("invalid JSON - garbage", base_url, "not even json")
    # 3. Empty body
    test("empty body", base_url, None)
    # 4. Empty JSON object
    test("empty JSON object", base_url, "{}")
    # 5. Empty reviews list
    test("empty reviews list", base_url, '{"reviews": []}')
    # 6. Missing reviews key
    test("missing reviews key", base_url, '{"something": 42}')
    # 7. Empty string text
    test("empty string text", base_url, '{"reviews": [{"id": 1, "text": ""}]}')
    # 8. Whitespace-only text
    test("whitespace-only text", base_url, '{"reviews": [{"id": 1, "text": "   "}]}')
    # 9. Wrong type for id (string instead of int)
    test("wrong type - id is string", base_url, '{"reviews": [{"id": "one", "text": "bad"}]}')
    # 10. Wrong type for text (int instead of string)
    test("wrong type - text is int", base_url, '{"reviews": [{"id": 1, "text": 123}]}')
    # 11. Null text
    test("null text", base_url, '{"reviews": [{"id": 1, "text": null}]}')
    # 12. Wrong content type with valid JSON
    test("wrong content type", base_url, '{"reviews": [{"id": 1, "text": "ok"}]}', content_type="text/plain")

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 40}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
