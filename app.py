
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.retriever import DocumentRetriever
from src.rag import answer_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="KnowFlow AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 KnowFlow AI")
st.caption(
    "Your private, local-first AI assistant for asking questions "
    "about PDF documents."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

defaults = {
    "retriever": None,
    "indexed_file_hash": None,
    "indexed_filename": None,
    "index_summary": None,
    "messages": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Document settings")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a text-based PDF to start asking questions.",
    )

    st.divider()

    st.subheader("How it works")
    st.markdown(
        """
        1. Upload a PDF.
        2. Extract and chunk its text.
        3. Find relevant passages using semantic search.
        4. Generate an answer with your local Ollama model.
        5. Review the source pages.
        """
    )

    st.divider()

    st.caption("Language model: llama3.2:1b")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector search: FAISS")
    st.caption("Interface: Streamlit")


# --------------------------------------------------
# Index uploaded PDF only when its content changes
# --------------------------------------------------

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_hash != st.session_state.indexed_file_hash:
        # Do not accidentally use results from the previous document.
        st.session_state.retriever = None
        st.session_state.indexed_file_hash = None
        st.session_state.indexed_filename = None
        st.session_state.index_summary = None
        st.session_state.messages = []

        temp_path = None

        try:
            with st.spinner(
                "Reading PDF and creating document embeddings..."
            ):
                # The existing ingestion pipeline accepts a file path.
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".pdf",
                    delete=False,
                ) as temp_file:
                    temp_file.write(pdf_bytes)
                    temp_path = Path(temp_file.name)

                retriever = DocumentRetriever()

                summary = retriever.index_pdf(
                    temp_path,
                    chunk_size=500,
                    overlap=50,
                )

                # Save the successfully indexed document in this session.
                st.session_state.retriever = retriever
                st.session_state.indexed_file_hash = current_hash
                st.session_state.indexed_filename = uploaded_file.name
                st.session_state.index_summary = summary

            st.success("PDF indexed successfully!")

        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


# --------------------------------------------------
# Document status
# --------------------------------------------------

if st.session_state.retriever is not None:
    summary = st.session_state.index_summary

    st.success(
        f"Ready to answer questions about "
        f"**{st.session_state.indexed_filename}**"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF pages",
        summary["total_pages"],
    )

    col2.metric(
        "Text chunks",
        summary["total_chunks"],
    )

    col3.metric(
        "Embedding dimensions",
        summary["embedding_dimension"],
    )

    st.divider()

    st.subheader("Ask your document")

    st.caption(
        "Ask a question based on the uploaded PDF. "
        "Answers may be unavailable if the document does not contain "
        "enough relevant information."
    )

    # Display the conversation.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("View source pages"):
                        for source in sources:
                            st.markdown(
                                f"- **{source['source']}** — "
                                f"Page {source['page_number']}"
                            )
                else:
                    st.caption("No supporting source pages found.")

    # Process the next question.
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the document and generating an answer..."
            ):
                try:
                    result = answer_question(
                        question,
                        st.session_state.retriever,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("View source pages"):
                            for source in sources:
                                st.markdown(
                                    f"- **{source['source']}** — "
                                    f"Page {source['page_number']}"
                                )
                    else:
                        st.caption(
                            "No supporting source pages found."
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    st.error(
                        f"Could not generate an answer: {exc}"
                    )

else:
    st.info(
        "Upload a PDF using the sidebar to get started."
    )

    st.markdown(
        """
        ### What you can do with KnowFlow AI

        - **Ask questions:** Find specific information in a document.
        - **Get grounded answers:** Use retrieved document passages.
        - **Check sources:** See which pages support an answer.
        - **Keep processing local:** Use your local Ollama model
          for answer generation.
        """
    )

    st.caption(
        "Note: Scanned PDFs containing only images are not supported "
        "yet because OCR has not been implemented."
    )
