
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
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Professional dark navy theme
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --bg: #080d19;
        --panel: #101827;
        --panel-soft: #141f32;
        --border: #26334a;
        --text: #e8eef9;
        --muted: #91a0b8;
        --accent: #63b3ff;
        --accent-soft: #18385a;
        --success: #39d9a4;
    }
    
    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(36, 87, 153, 0.17),
                transparent 36%
            ),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 25, 0.8);
    }

    [data-testid="stSidebar"] {
        background: #0c1321;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #245fba, #44b6df);
        font-size: 23px;
        box-shadow: 0 8px 24px rgba(48, 130, 220, 0.2);
    }

    .brand-name {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #f3f7ff;
    }

    .brand-caption {
        font-size: 0.8rem;
        color: var(--muted);
        margin-bottom: 1.3rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: clamp(1.9rem, 4vw, 2.7rem);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: #f1f5ff;
        margin-bottom: 0.6rem;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 720px;
        margin-bottom: 1.6rem;
    }

    .section-title {
        color: #eef4ff;
        font-size: 1.1rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 1rem;
    }

    .status-card {
        background: linear-gradient(
            135deg,
            rgba(21, 39, 63, 0.95),
            rgba(15, 25, 42, 0.95)
        );
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 16px;
        margin: 8px 0 16px 0;
    }

    .status-label {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .status-value {
        color: #f0f5ff;
        font-size: 0.97rem;
        font-weight: 700;
        margin-top: 7px;
        overflow-wrap: anywhere;
    }

    .status-dot {
        color: var(--success);
        font-size: 0.8rem;
    }

    .step-row {
        display: flex;
        gap: 11px;
        align-items: flex-start;
        margin: 15px 0;
    }

    .step-number {
        flex-shrink: 0;
        width: 29px;
        height: 29px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 9px;
        background: var(--accent-soft);
        color: #8dcaff;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .step-heading {
        color: #eaf1ff;
        font-size: 0.87rem;
        font-weight: 700;
    }

    .step-detail {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 3px;
    }

    .feature-card {
        background: rgba(16, 24, 39, 0.85);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 17px;
        height: 100%;
    }

    .feature-icon {
        font-size: 1.45rem;
        margin-bottom: 9px;
    }

    .feature-heading {
        color: #edf4ff;
        font-weight: 750;
        font-size: 0.96rem;
        margin-bottom: 5px;
    }

    .feature-copy {
        color: var(--muted);
        font-size: 0.83rem;
        line-height: 1.6;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        padding-top: 1rem;
        color: var(--muted);
        font-size: 0.75rem;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: rgba(16, 24, 39, 0.9);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: #edf5ff;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border-color: var(--border);
    }

    .stChatMessage {
        border: 1px solid rgba(57, 73, 99, 0.5);
        border-radius: 15px;
        padding: 13px;
        background: rgba(16, 24, 39, 0.55);
    }

    [data-testid="stChatInput"] {
        border-color: #344967;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(16, 24, 39, 0.6);
    }

    .stCaption {
        color: var(--muted);
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 768px) {
        .hero-title {
            letter-spacing: -0.5px;
        }

        .brand-name {
            font-size: 1.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    "index_error": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_document_state():
    """Clear indexed data and chat history from the current session."""
    st.session_state.retriever = None
    st.session_state.indexed_file_hash = None
    st.session_state.indexed_filename = None
    st.session_state.index_summary = None
    st.session_state.messages = []
    st.session_state.index_error = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">📚</div>
            <div class="brand-name">KnowFlow AI</div>
        </div>
        <div class="brand-caption">
            Private document intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your knowledge base")
    st.caption("Upload a PDF to start a private conversation with it.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Text-based PDFs are supported. Image-only scanned PDFs need OCR.",
        key="pdf_uploader",
    )

    st.divider()

    st.markdown("### Document status")

    if st.session_state.retriever is not None:
        st.markdown(
            '<span class="status-dot">●</span> Document indexed',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Active document</div>
                <div class="status-value">
                    {st.session_state.indexed_filename
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Clear conversation",
            use_container_width=True,
            help="Remove chat messages while keeping the indexed PDF.",
        ):
            st.session_state.messages = []
            st.rerun()

        if st.button(
            "Reset indexed document",
            use_container_width=True,
            help="Remove the current index from this session.",
        ):
            reset_document_state()
            st.rerun()

    elif st.session_state.index_error:
        st.error("Document indexing failed.")
        st.caption(st.session_state.index_error)

    else:
        st.caption("No document indexed yet.")

    st.divider()

    st.markdown("### How it works")

    steps = [
        ("01", "Upload", "Select your PDF document."),
        ("02", "Understand", "Extract text and create embeddings."),
        ("03", "Retrieve", "Find relevant passages with FAISS."),
        ("04", "Answer", "Generate a grounded response locally."),
        ("05", "Verify", "Review the cited source pages."),
    ]

    for number, heading, detail in steps:
        st.markdown(
            f"""
            <div class="step-row">
                <div class="step-number">{number}</div>
                <div>
                    <div class="step-heading">{heading}</div>
                    <div class="step-detail">{detail}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Local AI stack")
    st.caption("◉ Ollama · llama3.2:1b")
    st.caption("◉ Sentence Transformers")
    st.caption("◉ FAISS vector search")
    st.caption("◉ Streamlit interface")

    st.markdown(
        '<div class="footer">LOCAL-FIRST BY DESIGN<br>'
        'Your PDF is processed by this application. '
        'LLM generation uses your configured Ollama service.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Main header
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">PRIVATE DOCUMENT WORKSPACE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Your documents. '
    'Your answers.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-copy">
        Ask questions in natural language. KnowFlow AI searches your
        document, generates an evidence-grounded response and shows
        you which pages support the answer.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Index uploaded PDF only when its content changes
# --------------------------------------------------

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_hash != st.session_state.indexed_file_hash:
        reset_document_state()

        temp_path = None

        try:
            with st.spinner(
                "Reading PDF, extracting text and building the index..."
            ):
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

                st.session_state.retriever = retriever
                st.session_state.indexed_file_hash = current_hash
                st.session_state.indexed_filename = uploaded_file.name
                st.session_state.index_summary = summary
                st.session_state.index_error = None

            st.toast("PDF indexed successfully!", icon="✅")

        except Exception as exc:
            reset_document_state()
            st.session_state.index_error = str(exc)
            st.error(
                "Could not index this PDF. Check that it contains "
                "extractable text and try again."
            )
            with st.expander("Technical error details"):
                st.code(str(exc))

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


# --------------------------------------------------
# Document status and metrics
# --------------------------------------------------

if st.session_state.retriever is not None:
    summary = st.session_state.index_summary

    st.success(
        f"Ready to answer questions about "
        f"**{st.session_state.indexed_filename}**"
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "PDF pages",
        summary["total_pages"],
        help="Number of pages extracted by the PDF ingestion pipeline.",
    )

    metric2.metric(
        "Indexed chunks",
        summary["total_chunks"],
        help="Number of text chunks indexed for retrieval.",
    )

    metric3.metric(
        "Embedding dimensions",
        summary["embedding_dimension"],
        help="Dimensions of the document embedding vectors.",
    )

    st.divider()

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">Ask your document</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-copy">'
            'Get concise answers grounded in the indexed PDF.'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-copy">'
            'MODEL · llama3.2:1b &nbsp;|&nbsp; '
            'RETRIEVAL · FAISS'
            '</div>',
            unsafe_allow_html=True,
        )

    # Display existing conversation.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander(
                        f"📎 Sources used ({len(sources)})"
                    ):
                        for source in sources:
                            source_name = source.get(
                                "source", "Unknown source"
                            )
                            page_number = source.get(
                                "page_number", "Unknown"
                            )

                            st.markdown(
                                f"**{source_name}** · "
                                f"Page {page_number}"
                            )
                else:
                    st.caption("No supporting source pages returned.")

    # New question input.
    question = st.chat_input(
        "Ask something about your PDF...",
        max_chars=2000,
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
                "Searching relevant passages and generating an answer..."
            ):
                try:
                    result = answer_question(
                        question,
                        st.session_state.retriever,
                    )

                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(
                            f"📎 Sources used ({len(sources)})"
                        ):
                            for source in sources:
                                st.markdown(
                                    f"**{source.get('source', 'Unknown source')}** "
                                    f"· Page {source.get('page_number', 'Unknown')}"
                                )
                    else:
                        st.caption(
                            "No supporting source pages were returned. "
                            "The answer may be unavailable in the document."
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
                        "Could not generate an answer. Check the hosted API configuration or local Ollama service."
                    )

                    with st.expander("Technical error details"):
                        st.code(str(exc))

    if st.session_state.messages:
        conversation_text = []

        for message in st.session_state.messages:
            role = (
                "You"
                if message["role"] == "user"
                else "KnowFlow AI"
            )
            conversation_text.append(
                f"{role}:\n{message['content']}"
            )

        st.download_button(
            "Export conversation as TXT",
            data="\n\n".join(conversation_text),
            file_name="knowflow_conversation.txt",
            mime="text/plain",
        )

else:
    # Empty state.
    st.markdown(
        """
        <div class="status-card">
            <div class="eyebrow">GET STARTED</div>
            <div class="section-title">
                Bring your knowledge into the conversation
            </div>
            <div class="feature-copy">
                Upload a PDF from the sidebar. KnowFlow AI will extract
                its text, create searchable embeddings and prepare
                a private question-answering workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature1, feature2, feature3 = st.columns(3, gap="medium")

    with feature1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-heading">Semantic search</div>
                <div class="feature-copy">
                    Find relevant passages by meaning, not just
                    exact keyword matches.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-heading">Grounded answers</div>
                <div class="feature-copy">
                    Generate answers using retrieved document
                    context and a local language model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-heading">Source references</div>
                <div class="feature-copy">
                    Inspect the source filename and page references
                    returned for each answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Scanned PDFs containing only images are not supported yet. "
        "OCR has not been implemented in the current ingestion pipeline."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <strong>KNOWFLOW AI</strong> &nbsp;·&nbsp;
        LOCAL-FIRST PDF INTELLIGENCE<br>
        Built with Streamlit, FAISS, Sentence Transformers and Ollama.
        Answers can still be incomplete or incorrect; verify important
        information against the original document.
    </div>
    """,
    unsafe_allow_html=True,
)
