"""Document parser utilities."""
from __future__ import annotations

from pathlib import Path

import fitz

def parse_pdf(pdf_path: str) -> str:
    """Parse a PDF at `pdf_path` and return full text."""
    path = Path(pdf_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError("File does not exist or is not a PDF")

    text_parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "".join(text_parts)
