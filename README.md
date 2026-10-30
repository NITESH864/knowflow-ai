# KnowFlow AI

**KnowFlow AI** is a Streamlit-based PDF question-answering application
built around Retrieval-Augmented Generation (RAG). Upload a PDF, ask
questions about its contents, and receive answers grounded in retrieved
document context with source references.

Repository:
[NITESH864/knowflow-ai](https://github.com/NITESH864/knowflow-ai)

## Live Demo

Try KnowFlow AI here:

**[Launch KnowFlow AI](https://knowflow-ai-ksbvcp9po57tmygg7ysfkb.streamlit.app/)**

Upload a PDF, ask questions about its content, and review the generated answers and available source references.

## Features

-   Ask questions about an uploaded PDF.
-   Extract and chunk PDF text for retrieval.
-   Generate semantic embeddings with Sentence Transformers.
-   Retrieve relevant chunks using FAISS.
-   Generate answers using Groq-hosted inference or local Ollama
    inference.
-   Display source information and page references where available.
-   Run automated tests and a RAG evaluation script.

## How it works

1.  Read the uploaded PDF.
2.  Split extracted text into chunks.
3.  Embed the chunks using `all-MiniLM-L6-v2`.
4.  Index and retrieve relevant chunks with FAISS.
5.  Send the retrieved context and question to the configured language
    model.
6.  Display the answer and available source references.

``` text
PDF upload
   -> PDF text extraction
   -> Text chunking
   -> Sentence Transformers embeddings
   -> FAISS retrieval <- User question
   -> Retrieved document context
   -> Groq (cloud) or Ollama (local)
   -> Answer with source references
```

## Tech stack

-   Python
-   Streamlit
-   PyMuPDF for PDF text extraction
-   Sentence Transformers for embeddings
-   FAISS CPU for vector search
-   Requests for API calls
-   Groq API for hosted inference
-   Ollama for local inference
-   pytest for automated tests

The current embedding model is `all-MiniLM-L6-v2`. The default local
Ollama model is `llama3.2:1b`. The default Groq model in the client is
`openai/gpt-oss-20b`; availability depends on your Groq account and
provider configuration.

## Repository structure

``` text
knowflow-ai/
├── app.py
├── llm_client.py
├── src/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── pdf_reader.py
│   ├── rag.py
│   ├── retriever.py
│   └── vector_store.py
├── tests/
├── data/
│   └── sample_policy.pdf
├── evaluate_rag.py
├── requirements.txt
├── README.md
└── .gitignore
```

The repository may also contain evaluation, debugging, and manual test
scripts.

## Getting started

### 1. Clone the repository

``` bash
git clone https://github.com/NITESH864/knowflow-ai.git
cd knowflow-ai
```

### 2. Create a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the language model

KnowFlow AI supports Groq-hosted inference and local Ollama inference
through `llm_client.py`.

#### Option A: Groq API

For Streamlit Community Cloud, open the app's **Settings → Secrets** and
add:

``` toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Replace the example value with your own API key. Never commit API keys
or place them directly in source code. Make sure the selected model is
available to your Groq account.

The client uses Groq when `GROQ_API_KEY` is available.

#### Option B: Ollama (local inference)

1.  Install and start [Ollama](https://ollama.com/).

2.  Pull the default model:

    ``` bash
    ollama pull llama3.2:1b
    ```

3.  Ensure Ollama is running locally before starting the app.

When no Groq API key is configured, the current client falls back to
`http://localhost:11434/api/generate`. This fallback requires Ollama to
be running on the same machine as the app. A cloud-hosted app cannot use
`localhost` to reach Ollama on your personal computer.

### 5. Run the app

``` bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Tests and evaluation

Run the test suite:

``` bash
python -m pytest -q
```

Run the RAG evaluation:

``` bash
python evaluate_rag.py
```

### Latest recorded checks

  Check                            Result
  ----------------- ---------------------
  Automated tests               38 passed
  RAG evaluation      10/10 checks passed

These are the latest recorded local results and may change after code or
configuration changes.

## Deployment

The project has been configured for Streamlit Community Cloud with
Groq-hosted inference.

To deploy your own copy:

1.  Push the project to GitHub.
2.  Create an app in [Streamlit Community
    Cloud](https://share.streamlit.io/) and select `app.py` as the entry
    point.
3.  Add `GROQ_API_KEY` and optionally `GROQ_MODEL` under **Settings →
    Secrets**.
4.  Deploy or reboot the app after changing secrets.

**Live demo:** Add the exact Streamlit Community Cloud URL here once
confirmed. No demo URL is included because the exact public URL was not
available for this README.

## Security and limitations

-   Keep API keys in Streamlit Secrets or environment variables. Never
    commit credentials.
-   Cloud deployments cannot access an Ollama server running on your
    personal computer through `localhost`; use hosted inference for
    cloud deployment.
-   Answers depend on the uploaded PDF, retrieval quality, and model
    output. Check important information against the original document.
-   Source and page references depend on PDF text extraction and the
    document-processing pipeline.

## Future improvements

-   Expand the evaluation set with more questions and documents.
-   Measure retrieval improvements against a fixed evaluation set.
-   Improve loading and error states in the interface.
-   Add a verified public demo URL and screenshots.

## Author

**Nitesh Gupta**

-   GitHub: [@NITESH864](https://github.com/NITESH864)

## License

A license has not been specified in this README. Add a `LICENSE` file
and update this section if you choose to publish the project under a
particular license.
