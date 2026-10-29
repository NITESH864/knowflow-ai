
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"


def generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

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
        "options": {
            "temperature": 0
        },
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
    question = """
You are helping a computer science student.

RAG means Retrieval-Augmented Generation.

Explain these three steps:
1. Retrieval
2. Augmentation
3. Generation

Give one example using an employee policy PDF.
Do not discuss the red-yellow-green color system.
"""

    try:
        answer = generate_text(question)
        print("\nKnowFlow AI - LLM Test\n")
        print(answer)

    except (requests.RequestException, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
