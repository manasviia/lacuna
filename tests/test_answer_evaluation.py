"""
Tests for Stage 3: Answer Evaluation (LLM Judge).

All LLM and vector store calls are mocked. Tests cover:
  - Single answer evaluation: happy path, malformed JSON, invalid score
  - Full session evaluation: correct shape, gap flagging, caching to disk
  - Ordering enforcement: answers submitted before questions raises 409
  - Unknown question_id raises 400
  - FastAPI endpoint shape and error codes
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_judge_response(score: int = 2, rationale: str = "Good answer.") -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({"score": score, "rationale": rationale})
    return response


def _seed_questions(tmp_path: Path, session_id: str = "test123") -> list[dict]:
    questions = [
        {
            "id": f"q{i}",
            "concept": f"concept {i}",
            "text": f"Explain concept {i}.",
            "source_chunk_ids": ["0", "1"],
        }
        for i in range(1, 6)
    ]
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "questions.json").write_text(json.dumps(questions))
    return questions


def _make_mock_vector_store() -> MagicMock:
    chunk = MagicMock()
    chunk.page_content = "Relevant source content for evaluation."
    vs = MagicMock()
    vs.similarity_search.return_value = [chunk]
    return vs


def _make_answers(question_ids: list[str] | None = None) -> list[dict]:
    ids = question_ids or [f"q{i}" for i in range(1, 6)]
    return [{"question_id": qid, "text": f"Answer for {qid}."} for qid in ids]


# ---------------------------------------------------------------------------
# evaluate_answer() unit tests
# ---------------------------------------------------------------------------

class TestEvaluateAnswer:
    def test_returns_score_and_rationale(self):
        from app.evaluation.answer_evaluator import evaluate_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_judge_response(score=2, rationale="Mostly correct.")

        with patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm):
            result = evaluate_answer(
                question="What is X?",
                student_answer="X is ...",
                source_material="X refers to ...",
            )

        assert result["score"] == 2
        assert result["rationale"] == "Mostly correct."

    def test_accepts_all_valid_scores(self):
        from app.evaluation.answer_evaluator import evaluate_answer

        for score in (0, 1, 2, 3):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _make_judge_response(score=score)
            with patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm):
                result = evaluate_answer("Q?", "A.", "Source.")
            assert result["score"] == score

    def test_raises_on_malformed_json(self):
        from app.evaluation.answer_evaluator import evaluate_answer

        bad_response = MagicMock()
        bad_response.content = "Not JSON at all."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_response

        with patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ValueError, match="non-JSON"):
                evaluate_answer("Q?", "A.", "Source.")

    def test_raises_on_score_out_of_range(self):
        from app.evaluation.answer_evaluator import evaluate_answer

        bad_response = MagicMock()
        bad_response.content = json.dumps({"score": 5, "rationale": "too high"})
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_response

        with patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ValueError, match="invalid score"):
                evaluate_answer("Q?", "A.", "Source.")

    def test_raises_on_non_integer_score(self):
        from app.evaluation.answer_evaluator import evaluate_answer

        bad_response = MagicMock()
        bad_response.content = json.dumps({"score": "high", "rationale": "text score"})
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_response

        with patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ValueError, match="invalid score"):
                evaluate_answer("Q?", "A.", "Source.")

    def test_judge_uses_temperature_zero(self):
        """Evaluation calls must be deterministic (temperature=0)."""
        from app.evaluation.answer_evaluator import evaluate_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_judge_response()

        captured = {}

        def capture_llm(**kwargs):
            captured.update(kwargs)
            return mock_llm

        with patch("app.evaluation.answer_evaluator.ChatOpenAI", side_effect=capture_llm):
            evaluate_answer("Q?", "A.", "Source.")

        assert captured.get("temperature") == 0


# ---------------------------------------------------------------------------
# evaluate_all_answers() integration tests
# ---------------------------------------------------------------------------

class TestEvaluateAllAnswers:
    def test_returns_one_record_per_answer(self, tmp_path: Path):
        from app.evaluation.answer_evaluator import evaluate_all_answers

        _seed_questions(tmp_path)
        mock_vs = _make_mock_vector_store()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_judge_response(score=2)

        with (
            patch("app.evaluation.answer_evaluator.load_vector_store", return_value=mock_vs),
            patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm),
            patch("app.evaluation.answer_evaluator.settings") as mock_settings,
        ):
            mock_settings.judge_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.gap_score_threshold = 1
            mock_settings.session_log_dir = tmp_path

            results = evaluate_all_answers("test123", _make_answers())

        assert len(results) == 5

    def test_result_record_shape(self, tmp_path: Path):
        from app.evaluation.answer_evaluator import evaluate_all_answers

        _seed_questions(tmp_path)
        mock_vs = _make_mock_vector_store()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_judge_response(score=2)

        with (
            patch("app.evaluation.answer_evaluator.load_vector_store", return_value=mock_vs),
            patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm),
            patch("app.evaluation.answer_evaluator.settings") as mock_settings,
        ):
            mock_settings.judge_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.gap_score_threshold = 1
            mock_settings.session_log_dir = tmp_path

            results = evaluate_all_answers("test123", _make_answers())

        for r in results:
            assert "question_id" in r
            assert "concept" in r
            assert "score" in r
            assert "max_score" in r
            assert "rationale" in r
            assert "gap_flagged" in r
            assert r["max_score"] == 3

    def test_gap_flagging_at_threshold(self, tmp_path: Path):
        """Scores <= gap_score_threshold must set gap_flagged=True."""
        from app.evaluation.answer_evaluator import evaluate_all_answers

        _seed_questions(tmp_path)
        mock_vs = _make_mock_vector_store()

        # First two answers score 0 and 1 (both should be flagged at threshold=1)
        # Remaining three score 2 (should not be flagged)
        scores = [0, 1, 2, 2, 3]
        call_count = [0]

        def side_effect(*args, **kwargs):
            mock = MagicMock()
            mock.invoke.return_value = _make_judge_response(score=scores[call_count[0]])
            call_count[0] += 1
            return mock

        with (
            patch("app.evaluation.answer_evaluator.load_vector_store", return_value=mock_vs),
            patch("app.evaluation.answer_evaluator.ChatOpenAI", side_effect=side_effect),
            patch("app.evaluation.answer_evaluator.settings") as mock_settings,
        ):
            mock_settings.judge_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.gap_score_threshold = 1
            mock_settings.session_log_dir = tmp_path

            results = evaluate_all_answers("test123", _make_answers())

        flagged = [r for r in results if r["gap_flagged"]]
        assert len(flagged) == 2
        assert all(r["score"] <= 1 for r in flagged)

    def test_results_persisted_to_disk(self, tmp_path: Path):
        from app.evaluation.answer_evaluator import evaluate_all_answers

        _seed_questions(tmp_path)
        mock_vs = _make_mock_vector_store()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_judge_response(score=1)

        with (
            patch("app.evaluation.answer_evaluator.load_vector_store", return_value=mock_vs),
            patch("app.evaluation.answer_evaluator.ChatOpenAI", return_value=mock_llm),
            patch("app.evaluation.answer_evaluator.settings") as mock_settings,
        ):
            mock_settings.judge_model = "gpt-4o"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.gap_score_threshold = 1
            mock_settings.session_log_dir = tmp_path

            evaluate_all_answers("test123", _make_answers())

        answers_file = tmp_path / "test123" / "answers.json"
        assert answers_file.exists()
        saved = json.loads(answers_file.read_text())
        assert len(saved) == 5

    def test_raises_file_not_found_when_no_questions(self, tmp_path: Path):
        from app.evaluation.answer_evaluator import evaluate_all_answers

        with patch("app.evaluation.answer_evaluator.settings") as mock_settings:
            mock_settings.session_log_dir = tmp_path
            with pytest.raises(FileNotFoundError, match="No questions found"):
                evaluate_all_answers("nosession", _make_answers())

    def test_raises_key_error_for_unknown_question_id(self, tmp_path: Path):
        from app.evaluation.answer_evaluator import evaluate_all_answers

        _seed_questions(tmp_path)
        mock_vs = _make_mock_vector_store()

        with (
            patch("app.evaluation.answer_evaluator.load_vector_store", return_value=mock_vs),
            patch("app.evaluation.answer_evaluator.settings") as mock_settings,
        ):
            mock_settings.session_log_dir = tmp_path
            mock_settings.gap_score_threshold = 1

            with pytest.raises(KeyError, match="unknown question_id"):
                evaluate_all_answers(
                    "test123",
                    [{"question_id": "q99", "text": "Some answer."}],
                )


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestSubmitAnswersEndpoint:
    def _payload(self, question_ids=None):
        ids = question_ids or [f"q{i}" for i in range(1, 6)]
        return {
            "answers": [
                {"question_id": qid, "text": f"Answer for {qid}."}
                for qid in ids
            ]
        }

    def _fake_results(self, gap_at: list[int] | None = None) -> list[dict]:
        gap_at = gap_at or []
        return [
            {
                "question_id": f"q{i}",
                "concept": f"concept {i}",
                "score": 0 if i in gap_at else 2,
                "max_score": 3,
                "rationale": "Test rationale.",
                "gap_flagged": i in gap_at,
            }
            for i in range(1, 6)
        ]

    def test_returns_200_with_scores_and_gaps(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch("app.main.evaluate_all_answers", return_value=self._fake_results(gap_at=[2, 4])):
            client = TestClient(app)
            response = client.post("/sessions/abc12345/answers", json=self._payload())

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "abc12345"
        assert len(body["scores"]) == 5
        assert body["gap_count"] == 2
        assert len(body["gaps"]) == 2

    def test_returns_409_when_questions_not_generated(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch(
            "app.main.evaluate_all_answers",
            side_effect=FileNotFoundError("No questions found"),
        ):
            client = TestClient(app)
            response = client.post("/sessions/abc12345/answers", json=self._payload())

        assert response.status_code == 409

    def test_returns_400_for_unknown_question_id(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch(
            "app.main.evaluate_all_answers",
            side_effect=KeyError("unknown question_id 'q99'"),
        ):
            client = TestClient(app)
            response = client.post("/sessions/abc12345/answers", json=self._payload())

        assert response.status_code == 400

    def test_returns_404_for_unknown_session(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with patch(
            "app.main.evaluate_all_answers",
            side_effect=Exception("collection not found"),
        ):
            client = TestClient(app)
            response = client.post("/sessions/doesnotexist/answers", json=self._payload())

        assert response.status_code == 404
