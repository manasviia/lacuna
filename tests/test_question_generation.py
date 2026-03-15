"""
Tests for Stage 2: Question Generation.

All LLM and vector store calls are mocked — no API key required.
Tests cover:
  - Happy path question generation and shape validation
  - Caching behaviour (second call returns disk-persisted questions)
  - Malformed LLM response handling
  - Insufficient questions handling
  - FastAPI endpoint shape and error codes
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_chunks(n: int = 6) -> list:
    chunks = []
    for i in range(n):
        doc = MagicMock()
        doc.page_content = f"This is content for chunk {i} about concept {i}."
        doc.metadata = {"chunk_index": i, "session_id": "test123"}
        chunks.append(doc)
    return chunks


def _make_llm_response(questions: list[dict] | None = None) -> MagicMock:
    if questions is None:
        questions = [
            {"concept": f"concept {i}", "question": f"Explain concept {i} in detail."}
            for i in range(1, 6)
        ]
    response = MagicMock()
    response.content = json.dumps(questions)
    return response


# ---------------------------------------------------------------------------
# generate_questions() unit tests
# ---------------------------------------------------------------------------

class TestGenerateQuestions:
    def test_returns_five_questions(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = _make_llm_response()

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            questions = generate_questions("test123")

        assert len(questions) == 5

    def test_question_dict_shape(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = _make_llm_response()

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            questions = generate_questions("test123")

        for q in questions:
            assert "id" in q
            assert "concept" in q
            assert "text" in q
            assert "source_chunk_ids" in q
            assert isinstance(q["source_chunk_ids"], list)

    def test_question_ids_are_q1_through_q5(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = _make_llm_response()

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            questions = generate_questions("test123")

        assert [q["id"] for q in questions] == ["q1", "q2", "q3", "q4", "q5"]

    def test_questions_are_cached_to_disk(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = _make_llm_response()

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            generate_questions("test123")

        cache_file = tmp_path / "test123" / "questions.json"
        assert cache_file.exists()
        with open(cache_file) as f:
            cached = json.load(f)
        assert len(cached) == 5

    def test_second_call_uses_cache_not_llm(self, tmp_path: Path):
        """LLM should only be called once; second call reads from disk."""
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = _make_llm_response()

        settings_patch = {
            "top_k_retrieval": 6,
            "generation_model": "gpt-4o",
            "openai_api_key": "sk-fake",
            "session_log_dir": tmp_path,
        }

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            generate_questions("test123")
            generate_questions("test123")

        # LLM invoke should only have been called once
        assert mock_llm_instance.invoke.call_count == 1

    def test_raises_on_malformed_json(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        bad_response = MagicMock()
        bad_response.content = "This is not JSON at all."

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = bad_response

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            with pytest.raises(ValueError, match="non-JSON"):
                generate_questions("test123")

    def test_raises_when_fewer_than_five_questions_returned(self, tmp_path: Path):
        from app.generation.question_generator import generate_questions

        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = _make_mock_chunks()

        short_response = MagicMock()
        short_response.content = json.dumps([
            {"concept": "only one", "question": "Just one question."}
        ])

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = short_response

        with (
            patch("app.generation.question_generator.load_vector_store", return_value=mock_vs),
            patch("app.generation.question_generator.ChatOpenAI", return_value=mock_llm_instance),
            patch("app.generation.question_generator.settings") as mock_settings,
        ):
            mock_settings.top_k_retrieval = 6
            mock_settings.generation_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.session_log_dir = tmp_path

            with pytest.raises(ValueError, match="5 questions"):
                generate_questions("test123")


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestGetQuestionsEndpoint:
    def _fake_questions(self) -> list[dict]:
        return [
            {
                "id": f"q{i}",
                "concept": f"concept {i}",
                "text": f"Explain concept {i}.",
                "source_chunk_ids": ["0", "1"],
            }
            for i in range(1, 6)
        ]

    def test_returns_200_with_correct_shape(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch("app.main.generate_questions", return_value=self._fake_questions()):
            client = TestClient(app)
            response = client.get("/sessions/abc12345/questions")

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "abc12345"
        assert len(body["questions"]) == 5

    def test_returns_404_for_unknown_session(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch("app.main.generate_questions", side_effect=Exception("collection not found")):
            client = TestClient(app)
            response = client.get("/sessions/doesnotexist/questions")

        assert response.status_code == 404

    def test_returns_422_on_generation_value_error(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch("app.main.generate_questions", side_effect=ValueError("LLM returned non-JSON")):
            client = TestClient(app)
            response = client.get("/sessions/abc12345/questions")

        assert response.status_code == 422
