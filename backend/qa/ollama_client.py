# backend/qa/ollama_client.py
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes per request
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 2.0


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as e:
            last_err = e
            # Retry on timeouts or transient network issues
            if attempt < retries:
                sleep_for = backoff_seconds * (2 ** attempt)
                time.sleep(sleep_for)
                continue
            break

    raise RuntimeError(f"Ollama request failed after {retries+1} attempts: {last_err}")


def ollama_generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 512,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Uses Ollama /api/generate (non-streaming).
    """
    url = f"{base_url}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if system:
        payload["system"] = system

    out = _post_json(url, payload, timeout_seconds=timeout_seconds)
    # Ollama returns {"response": "...", ...}
    return out.get("response", "")


def ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    num_predict: int = 512,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Uses Ollama /api/chat (non-streaming).
    """
    url = f"{base_url}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "num_predict": num_predict,
    }

    out = _post_json(url, payload, timeout_seconds=timeout_seconds)
    # /api/chat returns {"message":{"content":"..."}, ...}
    msg = out.get("message") or {}
    return msg.get("content", "")
