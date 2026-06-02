"""
Claude client via Azure endpoint.
Handles the full message loop including tool use.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

from project_paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_ENDPOINT = os.getenv(
    "CLAUDE_ENDPOINT",
    "https://bhura-mhuqlv16-eastus2.services.ai.azure.com/anthropic/v1/messages",
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-5")
API_VERSION = os.getenv("CLAUDE_API_VERSION", "2024-05-01")


def call_claude(messages: list, system: str, tools: list = None,
                max_tokens: int = 4096, retries: int = 3) -> dict:
    """
    Call Claude Opus 4.5. Returns the raw API response dict.
    Retries on transient errors with exponential backoff.
    """
    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {CLAUDE_API_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    url = f"{CLAUDE_ENDPOINT.rstrip('/')}?api-version={API_VERSION}"

    for attempt in range(retries):
        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")
            if e.code in (429, 529) and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited (attempt {attempt+1}/{retries}), waiting {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Claude API error {e.code}: {detail[:500]}")
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Network error: {e}")

    raise RuntimeError("All retries exhausted")
