# KnowFlow AI 📚

**A private, local-first AI assistant for asking questions about PDF documents.**

KnowFlow AI uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from an uploaded PDF and generate answers grounded in that document. It is built with Streamlit, Ollama, sentence-transformers, and FAISS.

## Features

- **PDF ingestion:** Extracts text from PDF documents.
- **Text chunking:** Splits extracted content into smaller chunks for retrieval.
- **Semantic search:** Uses sentence embeddings and FAISS similarity search.
- **Hybrid retrieval:** Combines semantic and lexical relevance signals to improve retrieval.
- **Grounded answers:** Uses a local Ollama language model to answer questions using retrieved document context.
- **Source citations:** Returns the source PDF name and page numbers for supporting passages.
- **Unsupported-question handling:** Returns a clear fallback response when relevant information cannot be found.
- **Streamlit dashboard:** Dark-themed interface for uploading a PDF, indexing it, chatting with the document, and viewing document status.
- **Local-first workflow:** Designed to run on your own computer with a local model.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Language model runtime | Ollama |
| LLM | `llama3.2:1b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| PDF text extraction | PyMuPDF |
| Language | Python |
| Testing | pytest |

## How It Works

1. Upload a PDF through the Streamlit interface.
2. Extract text and split it into chunks while retaining page metadata.
3. Generate embeddings for the chunks.
4. Store and search the embeddings with FAISS.
5. Retrieve relevant context for the user's question.
6. Pass the retrieved context to the local language model.
7. Display the answer and available source-page citations.

```text
PDF Upload
    |
    v
Text Extraction (PyMuPDF)
    |
    v
Chunking + Page Metadata
    |
    v
Sentence-Transformer Embeddings
    |
    v
FAISS Vector Search + Retrieval
    |
    v
Relevant Context
    |
    v
Local Ollama LLM
    |
    v
Answer + Source Citations
```

## Prerequisites

- Windows, macOS, or Linux with Python installed
- [Ollama](https://ollama.com/) installed and running
- Git (optional, for cloning the repository)

Pull the configured model in Ollama:

```bash
ollama pull llama3.2:1b
```

## Installation

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username after publishing the repository.

```bash
git clone https://github.com/YOUR-USERNAME/knowflow-ai.git
cd knowflow-ai
```

If you already have the project locally, open a terminal in the project folder instead.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the [Python documentation](https://docs.python.org/3/library/venv.html) for your environment's activation policy.

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```powershell
python -m pip install -r requirements.txt
```

If there is no `requirements.txt` yet, create one from the project's verified dependencies before following this step. Do not install arbitrary package versions without checking the project's compatibility.

### 4. Start Ollama

Ensure Ollama is installed and running, and that the configured model is available:

```powershell
ollama list
ollama pull llama3.2:1b
```

### 5. Run the app

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Streamlit will print the local URL in the terminal, commonly `http://localhost:8501`.

## Usage

1. Start the application.
2. Upload a PDF using the file uploader.
3. Wait for indexing to complete.
4. Ask questions that can be answered from the document.
5. Review the answer and its source filename/page citations.

**Note:** The current workflow is designed for one indexed PDF at a time. Answers depend on the information available in the uploaded document and the passages retrieved for each question. Scanned PDFs may require OCR support if they do not contain extractable text.

## Testing

Run the automated tests:

```powershell
python -m pytest -q
```

Run the RAG evaluation:

```powershell
python evaluate_rag.py
```

### Last reported test results

- **Unit tests:** 38 passed
- **RAG evaluation:** 10/10 checks passed
- **Evaluation document:** `Tesla_Info.pdf`
- **Indexed pages:** 70
- **Indexed chunks:** 725

These are the results reported during development; rerun the commands after changes to verify the current state. Automated checks are useful but do not guarantee every generated answer is factually correct.

## Project Structure

The core files reported for this project include:

```text
knowflow-ai/
├── app.py
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── rag.py
│   └── vector_store.py
├── tests/
│   └── test_rag.py
├── debug_questions.py
├── evaluate_rag.py
└── README.md
```

Your exact folder contents may differ; keep this section aligned with the files actually present in your repository.

## Privacy and Limitations

- The project is designed around a local Ollama model and local document processing.
- Do not upload confidential PDFs unless you understand how every configured component handles data.
- Answer quality depends on PDF text extraction, retrieval quality, the selected model, and the source document.
- The application is a document-question-answering assistant, not a substitute for professional advice or independent fact-checking.

## Roadmap

Potential future improvements:

- Multi-PDF indexing and document selection
- More detailed source excerpts alongside page citations
- Improved answer formatting and factual consistency
- OCR support for scanned documents
- Additional evaluation questions and retrieval metrics
- Demo screenshots and a short project walkthrough

## Author

**Nitesh Gupta**  
Computer Science Engineering

---

If you find a bug or have a suggestion, open an issue in the repository after it is published.
