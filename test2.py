import fitz
from boundary_smart_splitter import StructureSplitter

doc = fitz.open("ASK.pdf")
text = ""
for page in doc:
    text += page.get_text()

splitter = StructureSplitter(chunk_size=1, max_chars=1500)
chunks = splitter.split(text)

import sys

print(f"Total chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    safe = chunk.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(safe)
    print()
