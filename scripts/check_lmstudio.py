"""Exercise the LM Studio OpenAI-compatible API with text and image requests."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI

from src.config.settings import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_FALLBACK_BASE_URLS,
    LM_STUDIO_IMAGE_MAX_TOKENS,
    LM_STUDIO_IMAGE_PROMPT,
    LM_STUDIO_MODEL,
    LM_STUDIO_TEMPERATURE,
    LM_STUDIO_TEXT_MAX_TOKENS,
    LM_STUDIO_TEXT_PROMPT,
    LM_STUDIO_TIMEOUT_S,
    TEST_IMAGE_PATH,
)


def list_models(base_url: str) -> list[str]:
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        timeout=LM_STUDIO_TIMEOUT_S,
    )
    response.raise_for_status()
    return [item["id"] for item in response.json()["data"]]


def try_base_url(base_url: str) -> bool:
    print(f"trying_base_url: {base_url}")
    try:
        models = list_models(base_url)
        print(f"models_found: {len(models)}")
        print(f"models_first_10: {', '.join(models[:10])}")
        model = LM_STUDIO_MODEL if LM_STUDIO_MODEL in models else models[0]
        print(f"model_used: {model}")

        client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=LM_STUDIO_TIMEOUT_S,
        )

        started = time.perf_counter()
        text_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": LM_STUDIO_TEXT_PROMPT}],
            temperature=LM_STUDIO_TEMPERATURE,
            max_tokens=LM_STUDIO_TEXT_MAX_TOKENS,
        )
        text_latency = time.perf_counter() - started
        print(f"text_reply: {text_response.choices[0].message.content}")
        print(f"text_latency_s: {text_latency:.2f}")

        image_base64 = base64.b64encode(TEST_IMAGE_PATH.read_bytes()).decode("ascii")
        started = time.perf_counter()
        image_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LM_STUDIO_IMAGE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            temperature=LM_STUDIO_TEMPERATURE,
            max_tokens=LM_STUDIO_IMAGE_MAX_TOKENS,
        )
        image_latency = time.perf_counter() - started
        print(f"image_reply: {image_response.choices[0].message.content}")
        print(f"image_latency_s: {image_latency:.2f}")
        return True
    except Exception as exc:
        print(f"failed: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    if not TEST_IMAGE_PATH.is_file():
        raise FileNotFoundError(TEST_IMAGE_PATH)

    for base_url in (LM_STUDIO_BASE_URL, *LM_STUDIO_FALLBACK_BASE_URLS):
        if try_base_url(base_url):
            print(f"working_base_url: {base_url}")
            return

    raise RuntimeError("All configured LM Studio base URLs failed")


if __name__ == "__main__":
    main()
