import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    groq_key = _get_secret("GROQ_API_KEY")

    if groq_key:
        groq_model = _get_secret(
            "GROQ_MODEL", DEFAULT_GROQ_MODEL
        )

        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": groq_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer accurately and clearly. "
                            "For document questions, use only "
                            "the supplied document context. "
                            "Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        answer = (
            result["choices"][0]["message"]["content"] or ""
        ).strip()

    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": (
                "Answer accurately and clearly. "
                "If unsure, do not invent facts."
            ),
            "stream": False,
            "options": {"temperature": 0},
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError("The model returned an empty answer")

    return answer


if __name__ == "__main__":
    question = "Explain RAG in one sentence."
    try:
        print(generate_text(question))
    except (requests.RequestException, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
