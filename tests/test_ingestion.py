"""
Tests for Stage 1: Document Ingestion Pipeline.

Design note: tests here are scoped to the purely deterministic parts
of the pipeline (chunker, pdf extractor error handling) and mock the
OpenAI and ChromaDB calls so the test suite runs without an API key
and without incurring cost. Integration tests that exercise the full
build_vector_store path belong in tests/integration/ (not yet added).
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Chunker tests — no external dependencies
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_returns_list_of_dicts(self):
        from app.ingestion.chunker import chunk_text

        chunks = chunk_text("Hello world. " * 100)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_dict_has_required_keys(self):
        from app.ingestion.chunker import chunk_text

        chunks = chunk_text("Some text. " * 60)
        for chunk in chunks:
            assert "content" in chunk
            assert "index" in chunk

    def test_indices_are_sequential_from_zero(self):
        from app.ingestion.chunker import chunk_text

        chunks = chunk_text("Paragraph.\n\n" * 80)
        indices = [c["index"] for c in chunks]
        assert indices == list(range(len(indices)))

    def test_empty_string_returns_empty_list(self):
        from app.ingestion.chunker import chunk_text

        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        from app.ingestion.chunker import chunk_text

        assert chunk_text("   \n\n\t  ") == []

    def test_short_text_produces_single_chunk(self):
        from app.ingestion.chunker import chunk_text

        text = "A short sentence."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["index"] == 0

    def test_long_text_produces_multiple_chunks(self):
        from app.ingestion.chunker import chunk_text

        # Generate text well beyond chunk_size (512 chars)
        text = "word " * 600
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_configured_size(self):
        from app.ingestion.chunker import chunk_text
        from app.config import settings

        text = "sentence. " * 400
        chunks = chunk_text(text)
        for chunk in chunks:
            # Allow a small buffer for the splitter's boundary logic
            assert len(chunk["content"]) <= settings.chunk_size + settings.chunk_overlap


# ---------------------------------------------------------------------------
# PDF extractor tests
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:
    def test_raises_on_image_only_pdf(self, tmp_path: Path):
        """A PDF with no extractable text layer should raise ValueError."""
        from app.ingestion.pdf_extractor import extract_text_from_pdf

        # Minimal valid PDF shell — no text content
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"xref\n0 3\n0000000000 65535 f \n"
            b"trailer\n<< /Root 1 0 R /Size 3 >>\nstartxref\n9\n%%EOF"
        )
        fake_pdf = tmp_path / "empty.pdf"
        fake_pdf.write_bytes(minimal_pdf)

        with pytest.raises(ValueError, match="No extractable text"):
            extract_text_from_pdf(fake_pdf)

    def test_page_markers_present_in_output(self, tmp_path: Path):
        """Extracted text should include [Page N] markers."""
        from app.ingestion.pdf_extractor import extract_text_from_pdf
        import pdfplumber

        # Mock pdfplumber.open to return a fake page with text
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page content."

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"placeholder")

        with patch("app.ingestion.pdf_extractor.pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(fake_pdf)

        assert "[Page 1]" in result
        assert "This is page content." in result


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestCreateSessionEndpoint:
    def test_rejects_non_pdf(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/sessions",
            files={"file": ("notes.txt", b"some text", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_successful_ingestion_returns_expected_shape(self, tmp_path: Path):
        """
        Full ingestion path with mocked PDF extraction, chunker, and
        vector store so no API key or ChromaDB is required.
        """
        from fastapi.testclient import TestClient
        from app.main import app

        fake_chunks = [{"content": f"chunk {i}", "index": i} for i in range(5)]

        with (
            patch("app.main.extract_text_from_pdf", return_value="some text"),
            patch("app.main.chunk_text", return_value=fake_chunks),
            patch("app.main.build_vector_store", return_value=MagicMock()),
        ):
            client = TestClient(app)
            response = client.post(
                "/sessions",
                files={"file": ("lecture.pdf", b"%PDF-fake", "application/pdf")},
            )

        assert response.status_code == 201
        body = response.json()
        assert "session_id" in body
        assert body["chunk_count"] == 5
        assert body["status"] == "ready"

    def test_empty_chunks_returns_422(self, tmp_path: Path):
        from fastapi.testclient import TestClient
        from app.main import app

        with (
            patch("app.main.extract_text_from_pdf", return_value="some text"),
            patch("app.main.chunk_text", return_value=[]),
            patch("app.main.build_vector_store", return_value=MagicMock()),
        ):
            client = TestClient(app)
            response = client.post(
                "/sessions",
                files={"file": ("lecture.pdf", b"%PDF-fake", "application/pdf")},
            )

        assert response.status_code == 422

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        assert client.get("/health").status_code == 200
