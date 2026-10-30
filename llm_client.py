import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value

    # Streamlit Community Cloud secrets.
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

    api_key = _get_secret("OPENAI_API_KEY")

    if api_key:
        hosted_model = _get_secret(
            "OPENAI_MODEL", DEFAULT_OPENAI_MODEL
        )

        response = requests.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": hosted_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant. "
                            "Answer accurately and clearly. "
                            "For document questions, use only "
                            "the supplied document context. "
                            "Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
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
                "You are a helpful AI assistant. "
                "Give accurate, clear answers. "
                "If you are unsure about a fact, "
                "say so instead of inventing information."
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
    question = (
        "Explain Retrieval-Augmented Generation and "
        "give one example using an employee policy PDF."
    )
    try:
        print(generate_text(question))
    except (requests.RequestException, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
