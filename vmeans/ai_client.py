"""
Small provider wrapper for the Hover Details AI assistant.

The UI sends one prepared prompt plus a compact data context.  This module
keeps provider-specific HTTP details out of the PyQt dialog.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List


class AIClientError(RuntimeError):
    """Raised when an AI provider cannot return a usable response."""


MODEL_PRESETS: Dict[str, List[str]] = {
    "Ollama": [
        "qwen2.5:14b-instruct",
        "llama3.2:latest",
        "llama3:8b",
        "mistral:latest",
        "gemma2:2b",
    ],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
    ],
    "Gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "Claude": [
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
    ],
}


def ask_ai(provider: str, model: str, system_prompt: str, user_prompt: str,
           timeout: int = 90) -> str:
    provider_key = provider.strip().lower()
    if provider_key == "ollama":
        return _ask_ollama(model, system_prompt, user_prompt, timeout)
    if provider_key == "openai":
        return _ask_openai(model, system_prompt, user_prompt, timeout)
    if provider_key == "gemini":
        return _ask_gemini(model, system_prompt, user_prompt, timeout)
    if provider_key == "claude":
        return _ask_claude(model, system_prompt, user_prompt, timeout)
    raise AIClientError(f"Unsupported AI provider: {provider}")


def _post_json(url: str, payload: Dict, headers: Dict[str, str],
               timeout: int) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise AIClientError(f"HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise AIClientError(f"Connection failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise AIClientError("Provider returned invalid JSON.") from exc


def _ask_ollama(model: str, system_prompt: str, user_prompt: str,
                timeout: int) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    result = _post_json(
        f"{host}/api/chat",
        payload,
        {"Content-Type": "application/json"},
        timeout,
    )
    content = result.get("message", {}).get("content", "")
    if not content:
        raise AIClientError("Ollama returned an empty response.")
    return content.strip()


def _ask_openai(model: str, system_prompt: str, user_prompt: str,
                timeout: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AIClientError("Set OPENAI_API_KEY to use OpenAI.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    result = _post_json(
        f"{base_url}/chat/completions",
        payload,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout,
    )
    choices = result.get("choices", [])
    if not choices:
        raise AIClientError("OpenAI returned no choices.")
    return choices[0].get("message", {}).get("content", "").strip()


def _ask_gemini(model: str, system_prompt: str, user_prompt: str,
                timeout: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AIClientError("Set GEMINI_API_KEY to use Gemini.")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {"temperature": 0.2},
    }
    result = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
    candidates = result.get("candidates", [])
    if not candidates:
        raise AIClientError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts).strip()


def _ask_claude(model: str, system_prompt: str, user_prompt: str,
                timeout: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIClientError("Set ANTHROPIC_API_KEY to use Claude.")
    payload = {
        "model": model,
        "max_tokens": 1200,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    result = _post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout,
    )
    chunks = result.get("content", [])
    text = "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")
    if not text:
        raise AIClientError("Claude returned an empty response.")
    return text.strip()
