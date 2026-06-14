"""test3.py – Demonstrate boundary-smart-splitter on xanatomy_proposal.pdf."""

import fitz
import sys
from boundary_smart_splitter import StructureSplitter


def extract_text(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def main() -> None:
    pdf_path = "xanatomy_proposal.pdf"
    text = extract_text(pdf_path)

    splitter = StructureSplitter(
        chunk_size=1,
        max_chars=1500,
        respect_headings=True,
        respect_numbered_sections=True,
        split_on_transitions=True,
    )

    chunks = splitter.split(text)

    print(f"File        : {pdf_path}")
    print(f"Total chars : {len(text)}")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        # Encode/decode to avoid Windows console encoding issues.
        safe = chunk.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)
        print(safe)
        print()


if __name__ == "__main__":
    main()
