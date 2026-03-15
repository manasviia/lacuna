from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.embedder import get_embedding_model


def build_vector_store(session_id: str, chunks: list[dict]) -> Chroma:
    """
    Embed all chunks for a session and persist them in a ChromaDB
    collection scoped to session_id.

    Each chunk is stored as a LangChain Document with metadata:
        - chunk_index: int   (position in the original chunked sequence)
        - session_id: str    (for cross-collection auditing if needed)

    The collection is named after session_id so that sessions are
    fully isolated; one corrupt or deleted session cannot affect others.

    Returns the Chroma instance for immediate use in the same request
    (avoids a redundant load call after build).
    """
    documents = [
        Document(
            page_content=chunk["content"],
            metadata={
                "chunk_index": chunk["index"],
                "session_id": session_id,
            },
        )
        for chunk in chunks
    ]

    persist_directory = str(settings.chroma_persist_dir / session_id)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        collection_name=session_id,
        persist_directory=persist_directory,
    )

    return vector_store


def load_vector_store(session_id: str) -> Chroma:
    """
    Load a previously persisted ChromaDB collection for a given session.

    Raises chromadb.errors.InvalidCollectionException if the session
    does not exist — callers should handle this and return a 404.
    """
    persist_directory = str(settings.chroma_persist_dir / session_id)

    return Chroma(
        collection_name=session_id,
        embedding_function=get_embedding_model(),
        persist_directory=persist_directory,
    )
