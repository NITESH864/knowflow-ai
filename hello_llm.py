import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2:1b",
    "prompt": (
        "Explain Retrieval-Augmented Generation (RAG) "
        "in simple English with one real-world example."
    ),
    "stream": False,
}

try:
    response = requests.post(
        url,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    print("\nLLM Response:\n")
    print(result["response"])

except requests.RequestException as error:
    print(f"Request failed: {error}")
