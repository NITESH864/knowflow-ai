
from src.ingestion import ingest_pdf


def test_ingest_sample_pdf():
    result = ingest_pdf(
        "data/sample_policy.pdf",
        chunk_size=60,
        overlap=10,
    )

    assert result["source"] == "sample_policy.pdf"
    assert result["total_pages"] == 2
    assert result["total_chunks"] == len(result["chunks"])
    assert result["total_chunks"] > 0

    page_numbers = {
        chunk["page_number"] for chunk in result["chunks"]
    }

    assert page_numbers == {1, 2}


def test_ingested_chunks_contain_policy_text():
    result = ingest_pdf(
        "data/sample_policy.pdf",
        chunk_size=200,
        overlap=20,
    )

    all_text = " ".join(
        chunk["text"] for chunk in result["chunks"]
    )

    assert "20 days of annual leave" in all_text
    assert "manager approval" in all_text
