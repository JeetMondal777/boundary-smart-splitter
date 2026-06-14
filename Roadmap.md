# ROADMAP — boundary-smart-splitter

> A context-aware, boundary-respecting text splitter for Python.
> LangChain-compatible. Framework-agnostic core.

---

## Vision

Most text splitters are **size-first** — they hit a character limit and cut, regardless of what they're cutting through. This library is **boundary-first** — it targets a size, but always lands on a clean linguistic boundary. The result is chunks that are more coherent, more embeddable, and more retrieval-friendly.

---

## Parameter Design Philosophy

Each splitter measures `chunk_size` in **its own semantic unit** — because words, sentences, and paragraphs are more meaningful units than raw characters. A `max_chars` ceiling acts as a universal safety net to protect embedding model context limits.

| Splitter | `chunk_size` unit | `max_chars` role |
|---|---|---|
| `WordSplitter` | N words | hard ceiling, never exceeded |
| `SentenceSplitter` | N sentences | if a sentence exceeds it, fall back to V1 |
| `ParagraphSplitter` | N paragraphs | if a paragraph exceeds it, fall back to V2 |
| `StructureSplitter` | N sections | if a section exceeds it, fall back to V3 |

**Fallback chain:** V4 → V3 → V2 → V1. Each version is a superset of the one before.

---

## Release Milestones

---

## ✅ V1 — Word-Boundary Splitter

**Goal:** Never cut a word in half. Always land on a space or punctuation boundary.

> `chunk_size` = **number of words** (default: 60 words ≈ ~300 chars average)
> `max_chars` = hard ceiling to protect embedding model limits (default: 500)

### Project Setup

- [ ] Initialize repo with proper Python package structure
  ```
  boundary-smart-splitter/
  ├── boundary_smart_splitter/
  │   ├── __init__.py
  │   ├── base.py          ← abstract base class
  │   ├── v1_word.py       ← V1 implementation
  │   └── langchain/
  │       ├── __init__.py
  │       └── compat.py    ← LangChain wrapper
  ├── tests/
  │   ├── test_v1.py
  │   └── fixtures/
  ├── examples/
  │   └── basic_usage.py
  ├── pyproject.toml
  ├── README.md
  ├── LICENSE
  └── ROADMAP.md
  ```
- [ ] Set up `pyproject.toml` with metadata, dependencies, and build config
- [ ] Add `MIT` license
- [ ] Set up GitHub repo with `.gitignore` for Python

### Core Logic

- [ ] Accept `chunk_size` (default: 60 words), `max_chars` (default: 500), `tolerance` (default: 10 words) as params
- [ ] Count words as you traverse — split when word count hits `chunk_size`
- [ ] If mid-word at split point → backtrack to last space
- [ ] If no space found within tolerance → extend forward to next space
- [ ] Enforce `max_chars` as absolute ceiling — never return a chunk exceeding it regardless of word count
- [ ] Strip leading/trailing whitespace from each chunk
- [ ] Handle edge cases: very long single words, empty strings, single-char inputs

### LangChain Compatibility

- [ ] Install `langchain-text-splitters` as optional dependency
- [ ] Create `LangChainWordSplitter` in `langchain/compat.py` that:
  - Inherits from `langchain_text_splitters.TextSplitter`
  - Overrides `split_text(text: str) -> List[str]`
  - Delegates to core `WordSplitter` internally
- [ ] Test that it works as a drop-in inside a LangChain RAG pipeline

### Testing

- [ ] Unit tests for normal text
- [ ] Unit tests for edge cases (single long word, empty string, whitespace-only)
- [ ] Test LangChain wrapper independently
- [ ] Test `max_chars` ceiling is never exceeded
- [ ] Test chunk word counts stay within `chunk_size + tolerance` range

### Publishing

- [ ] Write `README.md` with install instructions, quick-start code, and LangChain example
- [ ] Build package: `python -m build`
- [ ] Test on TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Publish to PyPI: `twine upload dist/*`
- [ ] Tag release `v1.0.0` on GitHub

---

## ✅ V2 — Sentence-Boundary Splitter

**Goal:** Never cut mid-sentence. Find the nearest `.` `?` or `!` forward or backward from the target.

> `chunk_size` = **number of sentences** (default: 5 sentences)
> `max_chars` = hard ceiling (default: 500) — if a single sentence exceeds it, fall back to V1 (word boundary) within that sentence

### Core Logic

- [ ] Extend `BaseSplitter` with `SentenceSplitter` class
- [ ] Count sentences as you traverse — split when sentence count hits `chunk_size`
- [ ] At target sentence boundary, scan **forward** first within tolerance for `.` `?` `!`
- [ ] If found → split there (include the punctuation in the chunk)
- [ ] If not found → scan **backward** to nearest sentence-ending punctuation within tolerance
- [ ] Rationale: forward-first keeps chunks fuller and more consistent; backward is the safety fallback
- [ ] If accumulated chars exceed `max_chars` before reaching `chunk_size` sentences → trigger V1 fallback
- [ ] Handle abbreviations that use periods: `e.g.`, `Dr.`, `U.S.A.` — do not split on these
  - Maintain a configurable abbreviation list
  - Use regex lookahead: punctuation followed by space + capital letter = real sentence end
- [ ] V2 inherits V1 behaviour — sentence boundary implies word boundary too
- [ ] Add `boundary_preference` param: `"forward"` (default) or `"backward"` (opt-in)

### LangChain Compatibility

- [ ] Create `LangChainSentenceSplitter` in `langchain/compat.py`
- [ ] Same interface pattern as V1 wrapper
- [ ] Ensure both V1 and V2 wrappers are exported from `boundary_smart_splitter.langchain`

### Testing

- [ ] Test sentence detection across `.` `?` `!`
- [ ] Test abbreviation handling does not cause false splits
- [ ] Test `forward` (default) vs `backward` preference produces different chunk boundaries
- [ ] Test `max_chars` ceiling triggers V1 fallback correctly
- [ ] Test mixed-punctuation paragraphs
- [ ] Test that LangChain wrapper produces identical output to core splitter

### Publishing

- [ ] Update `README.md` with V2 usage and comparison table
- [ ] Bump version to `v2.0.0` in `pyproject.toml`
- [ ] Re-build and re-publish to PyPI
- [ ] Tag release `v2.0.0` on GitHub
- [ ] Write LinkedIn post / changelog

---

## 🗂 Product Backlog

---

### V3 — Paragraph-Boundary Splitter *(Planned)*

Never break a paragraph (`\n\n`). Respects the author's own structural intent.

> `chunk_size` = **number of paragraphs** (default: 2 paragraphs)
> `max_chars` = hard ceiling — if a single paragraph exceeds it, fall back to V2 (sentence splitting) within that paragraph

**Key ideas:**
- Detect paragraph boundaries using `\n\n` (or configurable separator)
- Count paragraphs — split when count hits `chunk_size`
- If a single paragraph exceeds `max_chars`, fall back to V2 (sentence) splitting within it
- Add `overlap` param — carry the last sentence of the previous chunk into the next for context continuity
- Support Markdown paragraph detection as an optional mode

**Why it matters even without embeddings:**
Paragraph-level coherence improves BM25 and keyword retrieval too — not just vector search.

---

### V4 — Structure & Topic-Aware Splitter *(Planned)*

Respect document structure: headings, sections, and topic shifts.

> `chunk_size` = **number of sections/headings** (default: 1 section per chunk)
> `max_chars` = hard ceiling — if a section exceeds it, fall back to V3 (paragraph splitting) within it

**Key ideas:**
- Parse Markdown headings (`#`, `##`, `###`) as hard split boundaries
- Parse HTML structural tags (`<h1>`, `<section>`, `<article>`) as boundaries
- Detect implicit topic shifts via transition phrases: *"However"*, *"In contrast"*, *"Moving on"*
- Add `respect_headings: bool` param (default: `True`)
- Add `split_on_transition_phrases: bool` param (default: `False`)
- Hierarchical chunking mode: return `(chunk, metadata)` where metadata includes heading context
- If a section exceeds `max_chars` → fall back to V3 (paragraph), which falls back to V2, which falls back to V1

**Why V4 survives Vectorless RAG:**
Even in BM25, ColBERT, or LLM-native long-context pipelines, topic-coherent chunks produce better relevance signals. A chunk that starts mid-topic confuses any retrieval method — not just vector search. V4 is retrieval-method-agnostic by design.

---

## Design Principles

1. **Boundary-first, size-second** — clean boundaries always take priority over hitting exact counts
2. **Semantic units for `chunk_size`** — words count words, sentences count sentences, paragraphs count paragraphs
3. **`max_chars` as universal safety net** — protects embedding model limits without breaking the semantic contract
4. **Graceful fallback chain** — V4 → V3 → V2 → V1, each level catches what the level above can't handle
5. **LangChain-compatible, not LangChain-dependent** — the core never imports LangChain
6. **Each version is a superset** — V2 includes V1 behavior, V3 includes V2, and so on
7. **No ML dependencies in V1–V3** — fast, deterministic, offline-capable

---

## Quick API Preview

```python
# V1 — Word Splitter (chunk_size = number of words)
from boundary_smart_splitter import WordSplitter

splitter = WordSplitter(chunk_size=60, max_chars=500, tolerance=10)
chunks = splitter.split(text)

# V2 — Sentence Splitter (chunk_size = number of sentences)
from boundary_smart_splitter import SentenceSplitter

splitter = SentenceSplitter(chunk_size=5, max_chars=500, boundary_preference="forward")
chunks = splitter.split(text)

# LangChain-compatible (same params, LangChain interface)
from boundary_smart_splitter.langchain import LangChainSentenceSplitter

splitter = LangChainSentenceSplitter(chunk_size=5, max_chars=500)
chunks = splitter.split_text(text)  # LangChain interface
```

---

*Built by the community. Designed for the post-embedding era too.*