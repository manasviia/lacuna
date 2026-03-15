import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.vector_store import build_vector_store
from app.generation.question_generator import generate_questions
from app.evaluation.answer_evaluator import evaluate_all_answers
from pydantic import BaseModel

# Ensure storage directories exist at startup
settings.session_log_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

UPLOAD_TMP = Path("/tmp/lacuna_uploads")
UPLOAD_TMP.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Lacuna",
    description="RAG-based diagnostic tutoring API",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Stage 1: Document ingestion
# ---------------------------------------------------------------------------

@app.post("/sessions", status_code=201)
async def create_session(file: UploadFile = File(...)):
    """
    Ingest a PDF and initialize a session.

    Pipeline:
        PDF upload → text extraction → chunking → embedding → ChromaDB

    Returns session_id, chunk count, and status. The session_id is
    used as the key for all subsequent requests in this session.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Received: "
            f"'{file.filename or 'unnamed'}'.",
        )

    session_id = str(uuid.uuid4())[:8]
    tmp_path = UPLOAD_TMP / f"{session_id}_{file.filename}"

    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Text was extracted but produced zero usable chunks. "
                "Check that the PDF contains structured prose.",
            )

        build_vector_store(session_id, chunks)

        return JSONResponse(
            status_code=201,
            content={
                "session_id": session_id,
                "chunk_count": len(chunks),
                "status": "ready",
            },
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Stage 2: Question generation
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/questions")
def get_questions(session_id: str):
    """
    Generate (or retrieve cached) diagnostic questions for a session.

    On first call: retrieves diverse chunks from the session's vector store
    via MMR, sends them to the LLM, parses and persists the 5 questions.
    On subsequent calls: returns the persisted questions unchanged so that
    answer evaluation can reliably reference question text by ID.
    """
    try:
        questions = generate_questions(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Catches chromadb.errors.InvalidCollectionException and similar
        # "session not found" errors from the vector store layer.
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        ) from exc

    return {"session_id": session_id, "questions": questions}


# ---------------------------------------------------------------------------
# Stage 3: Answer submission and evaluation
# ---------------------------------------------------------------------------

class AnswerItem(BaseModel):
    question_id: str
    text: str


class AnswerSubmission(BaseModel):
    answers: list[AnswerItem]


@app.post("/sessions/{session_id}/answers")
def submit_answers(session_id: str, submission: AnswerSubmission):
    """
    Accept student answers and run LLM judge evaluation.

    Each answer is scored 0–3 by a separate judge call. Answers at or
    below GAP_SCORE_THRESHOLD are flagged as knowledge gaps. Results
    are persisted and returned immediately.

    Requires questions to have been generated first
    (GET /sessions/{id}/questions).
    """
    try:
        results = evaluate_all_answers(
            session_id,
            [a.model_dump() for a in submission.answers],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        ) from exc

    gaps = [r for r in results if r["gap_flagged"]]

    return {
        "session_id": session_id,
        "scores": results,
        "gap_count": len(gaps),
        "gaps": [r["concept"] for r in gaps],
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
