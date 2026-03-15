import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.ingestion.vector_store import load_vector_store


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

QUESTION_SYSTEM_PROMPT = """\
You are generating diagnostic questions to assess a student's conceptual \
understanding of academic material.

Your goal is to determine whether the student has understood key ideas — \
not whether they can reproduce specific sentences from the text.

Rules:
- Generate exactly 5 questions.
- Each question must target a distinct concept from the material.
- Each question must require a multi-sentence explanation to answer correctly. \
  Yes/no questions and single-word answers are not acceptable.
- Questions must be fully answerable from the provided material alone, \
  without outside knowledge.
- Distribute questions across different sections of the material. \
  Do not cluster them on the opening paragraph.

Return ONLY a valid JSON array with no surrounding text, markdown, or explanation:
[
  {
    "concept": "short label for the concept being tested (3–6 words)",
    "question": "the full question text as you would present it to a student"
  }
]
"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_questions(session_id: str) -> list[dict]:
    """
    Generate 5 diagnostic questions for a session and persist them.

    Retrieval uses Maximal Marginal Relevance (MMR) rather than
    top-k similarity so that the retrieved chunks cover distinct
    parts of the document. Without MMR, top-k queries against a
    generic "key concepts" string tend to cluster around the
    introduction, producing questions that all test the same material.

    Questions are cached to disk after the first call. Subsequent
    calls to this function return the same set, which is required
    for answer evaluation to reference question text by ID.

    Raises:
        ValueError: if the LLM returns malformed JSON or fewer than
                    5 questions, so the caller can decide how to surface this.
        chromadb.errors.InvalidCollectionException: if session_id does not
                    correspond to a built vector store (session not found).
    """
    questions_path = _questions_path(session_id)

    # Return cached questions if the session already has them
    if questions_path.exists():
        with open(questions_path) as f:
            return json.load(f)

    # Retrieve diverse chunks via MMR
    vector_store = load_vector_store(session_id)
    chunks = vector_store.max_marginal_relevance_search(
        query="key concepts, definitions, and main ideas",
        k=settings.top_k_retrieval,
        fetch_k=min(20, settings.top_k_retrieval * 4),
    )

    if not chunks:
        raise ValueError(
            f"Vector store for session '{session_id}' returned no chunks. "
            "The session may be empty or corrupted."
        )

    material = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    source_chunk_ids = [
        str(chunk.metadata.get("chunk_index", i)) for i, chunk in enumerate(chunks)
    ]

    # Generate questions via LLM
    llm = ChatOpenAI(
        model=settings.generation_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.4,  # some variability, but not so much that questions
                          # become inconsistent across re-runs of the same doc
    )

    response = llm.invoke([
        SystemMessage(content=QUESTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Material:\n\n{material}"),
    ])

    # Parse and validate
    try:
        raw = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON response for question generation. "
            f"Raw output: {response.content[:300]}"
        ) from exc

    if not isinstance(raw, list) or len(raw) < 5:
        raise ValueError(
            f"Expected a JSON array of 5 questions, got: {raw}"
        )

    questions = [
        {
            "id": f"q{i + 1}",
            "concept": item["concept"],
            "text": item["question"],
            # All questions share the same retrieved chunks since we
            # retrieve once for the full set. Per-question chunk attribution
            # is a future improvement (would require per-question retrieval).
            "source_chunk_ids": source_chunk_ids,
        }
        for i, item in enumerate(raw[:5])
    ]

    # Persist to session directory
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(questions_path, "w") as f:
        json.dump(questions, f, indent=2)

    return questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _questions_path(session_id: str) -> Path:
    """
    Questions are stored per-session under logs/{session_id}/questions.json.
    This directory is also where the full JSONL audit log will live (Stage 6),
    keeping all session artifacts co-located.
    """
    return settings.session_log_dir / session_id / "questions.json"
