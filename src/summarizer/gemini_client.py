from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx


class GeminiError(RuntimeError):
    """Raised when the Gemini API returns an error or non-JSON output."""


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _extract_json(text: str) -> str:
    t = text.strip()

    # Remove markdown fences if present
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line
        if lines:
            lines = lines[1:]
        # drop last fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    # If it already looks like JSON, still might have extra text after it.
    # Extract the first balanced JSON object/array.
    def first_balanced(s: str) -> str:
        # object
        if "{" in s:
            start = s.find("{")
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
        # array
        if "[" in s:
            start = s.find("[")
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
        return s.strip()

    extracted = first_balanced(t)
    return extracted.strip()



def gemini_generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",
    use_grounding: bool = False,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Call the Gemini API and return parsed JSON.

    Uses the Generative Language REST API (generativelanguage.googleapis.com).

    Env:
      - GEMINI_API_KEY: API key
      - GEMINI_MODEL: optional default model, e.g. 'gemini-2.0-flash'

    Grounding:
      If use_grounding=True, we add the Google Search tool in the request.
      Availability and billing depend on your Google plan.
    """
    api_key = _env(api_key_env)
    if not api_key:
        raise GeminiError(f"Missing {api_key_env} environment variable.")

    model_name = model or _env("GEMINI_MODEL", "gemini-2.0-flash")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    if use_grounding:
        payload["tools"] = [{"google_search": {}}]


    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)

    if resp.status_code >= 400:
        raise GeminiError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Extract model text. Gemini may return multiple parts (esp. with tools/grounding).
    try:
        parts = data["candidates"][0]["content"].get("parts", [])
        text = "\n".join(
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("text")
        ).strip()
    except Exception:
        raise GeminiError(f"Unexpected Gemini response format: {json.dumps(data)[:2000]}")


    # Parse JSON output (model asked to return JSON only)
    cleaned = _extract_json(text)
    # If grounding/tooling produced weird parts, try a last-resort JSON slice from the whole response
    if not cleaned:
        blob = json.dumps(data)
        l = blob.find("{")
        r = blob.rfind("}")
        if l != -1 and r != -1 and r > l:
            cleaned = blob[l : r + 1]

    try:
        return json.loads(cleaned)
    except Exception as e:
        raise GeminiError(
            f"Model did not return valid JSON. Error: {e}. "
            f"Raw: {text[:500]} | Cleaned: {cleaned[:500]}"
        )

