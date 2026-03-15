import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.ingestion.vector_store import load_vector_store


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are evaluating a student's free-text answer against source material \
from a course document. Your job is to assess conceptual understanding, \
not surface recall.

Score the answer on a 0–3 integer scale:

  3 — Complete and accurate. The student demonstrates clear understanding \
of the concept with no significant errors or gaps.
  2 — Mostly correct. The student understands the core idea but has minor \
gaps, imprecision, or omits secondary details.
  1 — Partial. The student shows some relevant understanding but misses \
key aspects, conflates concepts, or contains meaningful errors.
  0 — No understanding demonstrated. The answer is incorrect, irrelevant, \
or so vague it provides no evidence of comprehension.

Rules:
- Base your evaluation only on the provided source material. Do not penalize \
  a student for omitting information not present in the material.
- Be consistent: the same answer should receive the same score regardless \
  of phrasing.
- The rationale must be specific: cite what the answer got right, what it \
  missed, and why that affects the score.

Return ONLY a valid JSON object with no surrounding text or markdown:
{
  "score": <integer 0–3>,
  "rationale": "<one to three sentences explaining the score>"
}
"""


# ---------------------------------------------------------------------------
# Single-answer evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    student_answer: str,
    source_material: str,
) -> dict:
    """
    Score a single student answer using a dedicated LLM judge call.

    The judge is intentionally a separate call from the question generator.
    This allows the judge model to be swapped or benchmarked independently,
    and avoids the anchoring effect of the model evaluating output it
    effectively generated.

    Returns:
        {"score": int, "rationale": str}

    Raises:
        ValueError: if the judge returns malformed JSON or a score outside 0–3.
    """
    llm = ChatOpenAI(
        model=settings.judge_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,  # evaluations must be deterministic across runs
    )

    prompt = (
        f"Source material:\n{source_material}\n\n"
        f"Question:\n{question}\n\n"
        f"Student answer:\n{student_answer}"
    )

    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge returned non-JSON response: {response.content[:300]}"
        ) from exc

    score = result.get("score")
    if not isinstance(score, int) or score not in (0, 1, 2, 3):
        raise ValueError(
            f"Judge returned invalid score '{score}'. Expected integer 0–3."
        )

    return {
        "score": score,
        "rationale": result.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Full-session evaluation
# ---------------------------------------------------------------------------

def evaluate_all_answers(session_id: str, answers: list[dict]) -> list[dict]:
    """
    Evaluate all student answers for a session and persist the results.

    For each answer:
      1. Loads the corresponding question from the persisted question set.
      2. Retrieves the most relevant source chunk from ChromaDB using the
         question text as the query (similarity search against the session's
         vector store).
      3. Calls evaluate_answer() with the question, answer, and chunk.
      4. Flags answers at or below GAP_SCORE_THRESHOLD as knowledge gaps.

    Persists full results to logs/{session_id}/answers.json.

    Args:
        session_id: the session ID.
        answers: list of {"question_id": str, "text": str}

    Returns:
        list of evaluation records (see _build_record() for shape).

    Raises:
        FileNotFoundError: if questions have not yet been generated for
                           this session (caller should enforce ordering).
        KeyError: if an answer references a question_id not in the question set.
    """
    questions_path = _questions_path(session_id)
    if not questions_path.exists():
        raise FileNotFoundError(
            f"No questions found for session '{session_id}'. "
            "Call GET /sessions/{id}/questions before submitting answers."
        )

    with open(questions_path) as f:
        questions = {q["id"]: q for q in json.load(f)}

    vector_store = load_vector_store(session_id)

    results = []
    for answer in answers:
        qid = answer["question_id"]
        if qid not in questions:
            raise KeyError(
                f"Answer references unknown question_id '{qid}'. "
                f"Valid IDs: {list(questions.keys())}"
            )

        question_text = questions[qid]["text"]
        student_text = answer["text"]

        # Retrieve the single most relevant chunk for this question
        relevant_chunks = vector_store.similarity_search(question_text, k=1)
        source_material = (
            relevant_chunks[0].page_content if relevant_chunks else ""
        )

        evaluation = evaluate_answer(question_text, student_text, source_material)

        results.append({
            "question_id": qid,
            "concept": questions[qid]["concept"],
            "score": evaluation["score"],
            "max_score": 3,
            "rationale": evaluation["rationale"],
            "gap_flagged": evaluation["score"] <= settings.gap_score_threshold,
        })

    # Persist
    answers_path = _answers_path(session_id)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answers_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _questions_path(session_id: str) -> Path:
    return settings.session_log_dir / session_id / "questions.json"


def _answers_path(session_id: str) -> Path:
    return settings.session_log_dir / session_id / "answers.json"
