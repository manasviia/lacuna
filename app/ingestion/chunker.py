from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_text(text: str) -> list[dict]:
    """
    Split raw extracted text into overlapping chunks using recursive
    character splitting.

    Separator priority: paragraph breaks → line breaks → sentence
    boundaries → word boundaries → characters. This order is chosen
    to keep semantically coherent units together as long as they fit
    within chunk_size; falling back to finer splits only when necessary.

    Returns a list of dicts:
        [{"content": str, "index": int}, ...]

    Empty or whitespace-only chunks are filtered out before indexing
    so that chunk_index values in downstream metadata are dense and
    unambiguous.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    return [
        {"content": chunk, "index": i}
        for i, chunk in enumerate(c for c in raw_chunks if c.strip())
    ]
