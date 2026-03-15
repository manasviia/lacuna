# Lacuna

**Lacuna** is a RAG-based diagnostic tutoring system that ingests course material, probes a student's conceptual understanding through LLM-generated questions, and generates targeted follow-up questions for identified knowledge gaps — designed from the ground up to support rigorous empirical evaluation in educational research settings.

---

## Table of Contents

- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Evaluation & Future Work](#evaluation--future-work)
- [Setup](#setup)
- [Usage](#usage)
- [Motivation](#motivation)

---

## Architecture

The full pipeline from document ingestion to adaptive questioning:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ELENCHUS PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  PDF Upload  │  (lecture notes, textbook chapter, syllabus)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Chunking   │  Recursive character splitter, configurable chunk size
  │              │  and overlap; preserves paragraph boundaries
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Embedding   │  OpenAI text-embedding-3-small → dense vectors
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Vector Store │  ChromaDB (local, persistent collection per session)
  │  (ChromaDB)  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────┐
  │  Question Generation │  Top-k retrieved chunks → GPT-4o
  │  (5 diagnostic Qs)   │  Prompted to span concept breadth, not trivia
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Student Answers    │  Free-text input via FastAPI endpoint
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   LLM Judge          │  Separate evaluator call: scores each answer
  │   (Answer Eval)      │  0–3 rubric per question, returns rationale
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Gap Analysis       │  Aggregates low-scoring concepts;
  │                      │  maps scores back to source chunks
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Adaptive Follow-up  │  Targeted questions generated only for
  │  Question Generation │  concepts below threshold score
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Session Logger     │  Persists: input chunks, questions, answers,
  │   (JSONL + metadata) │  scores, rationales, model version, timestamps
  └──────────────────────┘
```

---

## Design Decisions

**Why RAG over fine-tuning.** The core challenge in a tutoring system is specificity to the material a student is actually studying — which changes every session. Fine-tuning a model on course content would require data collection pipelines, retraining infrastructure, and a lag between content availability and model readiness that is incompatible with a classroom setting. RAG solves this by treating the uploaded document as a live, queryable knowledge base. The tradeoff is that generation quality is bounded by retrieval quality, which is why chunking strategy and chunk size are treated as tunable parameters rather than constants. Retrieval recall on domain-specific academic text is a known weak point and is tracked explicitly in session logs.

**Why a separate LLM judge for evaluation.** Co-locating answer generation and answer evaluation in the same model call introduces a subtle but important confound: the model may unconsciously anchor its evaluation on the phrasing of its own prior output. More practically, separating the judge into a distinct call allows it to be swapped out independently — a smaller, cheaper model can be benchmarked against GPT-4o on the scoring task without touching the rest of the pipeline. The rubric (0–3 per question, with mandatory rationale) is intentionally simple enough to enable inter-rater reliability studies: human annotators can score the same answers and agreement rates can be computed, which is necessary if any downstream research is to be published.

**Why structured session logging matters for reproducibility.** One of the most persistent failure modes in applied ML research is the inability to reproduce results — not because the model changed, but because the exact prompts, retrieved chunks, and model version at inference time were never recorded. Every session in Lacuna writes a JSONL record containing the full prompt sent to the model (including system message), the raw retrieved chunks with their similarity scores, the student's raw input, each score with its evaluator rationale, and the OpenAI model string with API version. This makes it possible to re-run a session deterministically (modulo model stochasticity), diff behavior across model upgrades, and audit individual sessions if a student disputes a score. It is also the prerequisite for any meta-analysis across many student sessions.

**Why FastAPI over a notebook interface.** Jupyter notebooks are the natural prototyping environment for ML work, but they are difficult to instrument, do not enforce input/output contracts, and cannot be deployed in a controlled study environment. FastAPI enforces typed request and response schemas via Pydantic, making it straightforward to add authentication, rate limiting, and request logging without retrofitting. The API-first design also decouples the backend from whatever frontend a research team might want to pair it with — whether that is a simple web form, a learning management system integration, or a custom tablet interface for a classroom study.

---

## Evaluation & Future Work

**What a proper evaluation would require.** Measuring whether Lacuna actually improves learning outcomes requires an experimental design that most AI tutoring tools never attempt. The minimum credible study would be a randomized controlled trial in which students studying the same material are randomly assigned to: (a) the full Lacuna pipeline with adaptive follow-up, (b) Lacuna without the gap-analysis stage (all students receive the same follow-up questions regardless of performance), or (c) a control condition in which students self-study with no structured questioning. The primary outcome measure would be performance on a delayed retention test (48–72 hours post-session), not immediate recall, which conflates testing with learning. Secondary measures would include time-on-task, student-reported confidence calibration, and the correlation between Lacuna gap scores and actual test performance — the last of which functions as a direct validity check on the LLM judge.

**Known confounds and limitations.** The LLM judge's 0–3 rubric has not been validated against human expert scoring. Until inter-rater agreement (Cohen's kappa or Krippendorff's alpha) between the judge and domain experts is established, the scores should be treated as proxies rather than ground truth. Additionally, the quality of generated questions is bounded by the quality of the source document: a poorly structured syllabus will produce low-information questions. This suggests that a pre-processing quality filter — or a human review step before deployment in a study — would strengthen internal validity. Finally, all current evaluation of generation quality is qualitative; a benchmark set of (document, question, ideal-answer) triples would allow automated regression testing when models or prompts are updated.

**Planned extensions.** The session log schema is designed to support longitudinal analysis across multiple sessions for the same student. A natural next step is a spaced repetition layer that surfaces concepts from previous sessions that scored below threshold, weighted by time since last exposure. The gap analysis currently operates at the concept level inferred from chunk content; mapping gaps to a formal concept graph (e.g., derived from a course syllabus with prerequisite structure) would allow more principled identification of foundational versus terminal knowledge gaps. Finally, the evaluation pipeline is model-agnostic in principle but has only been tested with OpenAI models; extending to open-weight models would make it viable in privacy-sensitive institutional settings where API calls to external providers are restricted.

---

## Setup

**Requirements:** Python 3.11+, Docker (optional but recommended), an OpenAI API key.

### With Docker

```bash
git clone https://github.com/yourusername/lacuna.git
cd lacuna
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### Without Docker

```bash
git clone https://github.com/yourusername/lacuna.git
cd lacuna
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uvicorn app.main:app --reload
```

### Environment Variables

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=6
GAP_SCORE_THRESHOLD=1
SESSION_LOG_DIR=./logs
```

---

## Usage

### Start a session

Upload a PDF and receive a session ID:

```bash
curl -X POST http://localhost:8000/sessions \
  -F "file=@lecture_notes.pdf"
```

```json
{
  "session_id": "a3f9c12e",
  "chunk_count": 34,
  "status": "ready"
}
```

### Get diagnostic questions

```bash
curl http://localhost:8000/sessions/a3f9c12e/questions
```

```json
{
  "session_id": "a3f9c12e",
  "questions": [
    {
      "id": "q1",
      "text": "Explain how attention mechanisms in transformers differ from recurrent approaches to sequence modeling.",
      "source_chunk_ids": ["chunk_07", "chunk_12"]
    },
    ...
  ]
}
```

### Submit answers

```bash
curl -X POST http://localhost:8000/sessions/a3f9c12e/answers \
  -H "Content-Type: application/json" \
  -d '{
    "answers": [
      {"question_id": "q1", "text": "Attention allows the model to..."},
      ...
    ]
  }'
```

### Retrieve evaluation and follow-up questions

```bash
curl http://localhost:8000/sessions/a3f9c12e/results
```

```json
{
  "session_id": "a3f9c12e",
  "scores": [
    {
      "question_id": "q1",
      "score": 1,
      "max_score": 3,
      "rationale": "The answer identifies the correct mechanism but omits the role of positional encoding and does not address computational parallelism.",
      "gap_flagged": true
    },
    ...
  ],
  "follow_up_questions": [
    {
      "concept": "positional encoding",
      "text": "Why is explicit positional encoding necessary in a transformer architecture, and what would be lost without it?"
    },
    ...
  ]
}
```

### Session logs

Each session writes a JSONL record to `SESSION_LOG_DIR`:

```
logs/
  a3f9c12e.jsonl   # full prompt/response trace
  a3f9c12e.meta.json  # model versions, timestamps, config snapshot
```

---

## Motivation

The most common failure mode in educational AI tools is not technical — it is the absence of a theory of how to measure whether the tool works. Generic chatbots can answer questions about course material, but producing a plausible-sounding answer is not the same as diagnosing what a student does not yet understand and intervening at the right level of specificity. The research on intelligent tutoring systems going back to the 1980s (VanLehn, Anderson, Bloom's 2-sigma problem) is consistent on this point: the gains from one-on-one tutoring come not from access to information, but from tight feedback loops on misconceptions.

Lacuna is an attempt to operationalize that loop in a way that is both technically auditable and empirically testable. The design is deliberately constrained: it does not try to hold a free-form conversation, it does not try to explain concepts it has identified, and it does not try to predict long-term retention from a single session. It does one thing — identify where a student's understanding diverges from the material and generate pointed questions targeting that divergence — and it does it in a way that produces structured, loggable, reproducible outputs.

The broader challenge this project is oriented toward is the gap between AI systems that perform well on benchmarks and AI systems that can be deployed in real educational settings and evaluated with the same rigor we would expect from any other instructional intervention. That requires treating the engineering and the evaluation design as equally important problems. Lacuna is a starting point for working through both.

---

## License

MIT
