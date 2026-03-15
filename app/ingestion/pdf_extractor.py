from pathlib import Path

import pdfplumber


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract full text from a PDF file.

    Each page is prefixed with a [Page N] marker so that downstream
    chunking can preserve rough positional context. Pages with no
    extractable text (e.g. image-only pages) are silently skipped;
    if the entire document yields nothing, a ValueError is raised
    rather than returning empty text that would silently produce
    zero-information chunks.
    """
    text_pages: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_pages.append(f"[Page {i + 1}]\n{page_text.strip()}")

    if not text_pages:
        raise ValueError(
            f"No extractable text found in '{pdf_path.name}'. "
            "The file may be a scanned/image-only PDF. "
            "OCR support is not yet implemented."
        )

    return "\n\n".join(text_pages)
