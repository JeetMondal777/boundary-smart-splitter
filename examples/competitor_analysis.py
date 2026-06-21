import os
import sys
import statistics
import time
from typing import Any, Dict

# Ensure package is on the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boundary_smart_splitter import StructureSplitter, Evaluator
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import FakeEmbeddings

# Define test texts
WIKIPEDIA_TEXT = """# Python (programming language)
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.

## History
Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to ABC.

### Design philosophy
Python is a multi-paradigm programming language. Object-oriented programming and structured programming are fully supported.

## Syntax and semantics
Python is intended to be an easily readable language. Its formatting is visually uncluttered, and it often uses English keywords where other languages use punctuation.

### Indentation
Python uses whitespace indentation, rather than curly brackets or keywords, to delimit blocks. An increase in indentation comes after certain statements.
"""

ACADEMIC_TEXT = """[Page 1]
JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH, VOL. 88, 2026

Attention Is All You Need (Extracted Text)
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.

1. Introduction
Sequence transduction models, especially recurrent neural networks (RNN), have been widely established.
[Page 2]
However, recurrent models are fundamentally sequential in nature, which limits parallelization during training.

2. Background
The goal of reducing sequential computation also limits the efficiency of convolutional neural networks.
Copyright © 2026 JAIR. All rights reserved.
"""

NEWS_TEXT = """<h1>Breaking News: The Rise of AI Agents</h1>
<p>Published: June 2026</p>
<p>AI agents are transforming software engineering, automating tasks from code generation to deployment.</p>

<h2>How Agents Work</h2>
<p>By executing multi-turn tool calling, agents can resolve complex tasks autonomously.</p>

<h2>Future Outlook</h2>
<p>In summary, the transition from static systems to agentic systems is accelerating.</p>
"""

CODE_TEXT = """def split_text(text: str) -> list[str]:
    \"\"\"Split input text using a boundary-smart strategy.

    This function coordinates paragraph and sentence splitters to
    produce cohesive, retrieval-friendly chunks.
    \"\"\"
    if not text:
        return []
    
    # Process text paragraphs
    paragraphs = text.split("\\n\\n")
    return [p.strip() for p in paragraphs if p.strip()]
"""

DOCUMENTS = {
    "Wikipedia (Markdown)": WIKIPEDIA_TEXT,
    "Academic Paper (Dirty Text)": ACADEMIC_TEXT,
    "News Article (HTML)": NEWS_TEXT,
    "Python Code": CODE_TEXT
}


def check_boundaries(text: str, chunks: list[str]) -> float:
    """Return linguistic boundary score out of 10.0.
    
    10.0: all chunks end on sentence boundaries (. ! ? or newline).
    6.0: some chunks end mid-sentence, but all respect word boundaries (no cuts mid-word).
    0.0: any chunk cuts mid-word.
    """
    if not chunks:
        return 10.0
        
    # Check for mid-word cuts
    current_pos = 0
    for chunk in chunks:
        idx = text.find(chunk, current_pos)
        if idx == -1:
            idx = text.find(chunk)
        if idx != -1:
            if idx > 0:
                char_before = text[idx - 1]
                char_start = text[idx]
                if char_before.isalnum() and char_start.isalnum():
                    return 0.0
            end_idx = idx + len(chunk)
            if end_idx < len(text):
                char_end = text[end_idx - 1]
                char_after = text[end_idx]
                if char_end.isalnum() and char_after.isalnum():
                    return 0.0
            current_pos = end_idx

    # Check for mid-sentence cuts
    sentence_end_chars = ('.', '!', '?', '\n', '"', "'", ')', ']', '}')
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        if not stripped.endswith(sentence_end_chars):
            return 6.0
            
    return 10.0


def compute_score(metrics: dict, boundary_score: float) -> float:
    # 1. Topic Coherence (40 points)
    heading_pres = metrics["heading_preservation_rate"] * 20.0
    cross_topic = (1.0 - metrics["cross_topic_mixing_rate"]) * 20.0
    
    # 2. Sizing Safety (30 points)
    pct_under = (1.0 - metrics["pct_chunks_under_100_tokens"]) * 15.0
    pct_over = (1.0 - metrics["pct_chunks_over_512_tokens"]) * 15.0
    
    # 3. Speed (20 points)
    t = metrics["processing_time_ms"]
    if t <= 1.0:
        speed_score = 20.0
    elif t <= 5.0:
        speed_score = 18.0
    elif t <= 20.0:
        speed_score = 15.0
    elif t <= 100.0:
        speed_score = 10.0
    elif t <= 500.0:
        speed_score = 5.0
    else:
        speed_score = 2.0
        
    # 4. Boundary Accuracy (10 points)
    boundary_pts = boundary_score
    
    total = heading_pres + cross_topic + pct_under + pct_over + speed_score + boundary_pts
    return max(0.0, min(100.0, total))


def main():
    print("==========================================================")
    print("           COMPETITOR ANALYSIS & SCORING RUNNER           ")
    print("==========================================================\n")

    evaluator = Evaluator()
    
    # Initialize all 4 splitters
    splitters = {
        "StructureSplitter (Ours)": StructureSplitter(max_chars=1000, min_tokens=100),
        "RecursiveCharacterTextSplitter (LangChain)": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        "TokenTextSplitter (LangChain)": TokenTextSplitter(chunk_size=250, chunk_overlap=0),
        "SemanticChunker (LangChain experimental)": SemanticChunker(FakeEmbeddings(size=384))
    }

    # Gather metrics across all documents
    global_results = {s: [] for s in splitters}

    for doc_name, text in DOCUMENTS.items():
        print(f"Analyzing Document: {doc_name}...")
        for s_name, s_inst in splitters.items():
            # Time and Split
            start = time.perf_counter()
            if hasattr(s_inst, "split_text"):
                chunks = s_inst.split_text(text)
            else:
                chunks = s_inst.split(text)
            t_ms = (time.perf_counter() - start) * 1000.0

            # Evaluate metrics
            rep = evaluator.evaluate(text=text, splitter=s_inst)
            metrics = rep.to_dict()[type(s_inst).__name__]
            metrics["processing_time_ms"] = t_ms  # update with precise time

            # Calculate boundary accuracy
            b_score = check_boundaries(text, chunks)
            
            # Calculate final score
            score = compute_score(metrics, b_score)
            
            global_results[s_name].append({
                "score": score,
                "speed": t_ms,
                "coherence": (metrics["heading_preservation_rate"] + (1.0 - metrics["cross_topic_mixing_rate"])) / 2.0,
                "sizing": (1.0 - metrics["pct_chunks_under_100_tokens"] + 1.0 - metrics["pct_chunks_over_512_tokens"]) / 2.0,
                "boundary": b_score
            })

    # Compile average results
    summary_report = []
    summary_report.append("# Competitor Analysis & Quality Scoring Report\n")
    summary_report.append("This report scores splitters out of 100 based on Topic Coherence (40 pts), Sizing Safety (30 pts), Speed (20 pts), and Boundary Accuracy (10 pts) averaged across 4 diverse document formats.\n")
    
    table_header = "| Splitter | Overall Score (/100) | Topic Coherence (Avg) | Sizing Safety (Avg) | Avg Speed (ms) | Boundary Accuracy |"
    table_divider = "|:---|:---:|:---:|:---:|:---:|:---:|"
    table_rows = []

    print("\n======================= FINAL RESULTS =======================\n")
    for s_name in splitters:
        avg_score = statistics.mean([res["score"] for res in global_results[s_name]])
        avg_speed = statistics.mean([res["speed"] for res in global_results[s_name]])
        avg_coherence = statistics.mean([res["coherence"] for res in global_results[s_name]])
        avg_sizing = statistics.mean([res["sizing"] for res in global_results[s_name]])
        avg_boundary = statistics.mean([res["boundary"] for res in global_results[s_name]])
        
        row = f"| **{s_name.split(' (')[0]}** | **{avg_score:.1f}** | {avg_coherence:.1%} | {avg_sizing:.1%} | {avg_speed:.2f}ms | {avg_boundary:.1f}/10 |"
        table_rows.append(row)
        print(f"{s_name.split(' (')[0]}: {avg_score:.1f}/100")

    summary_report.append(table_header)
    summary_report.append(table_divider)
    summary_report.extend(table_rows)

    # Save to file
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competitor_analysis_results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_report))
    print(f"\nSaved detailed analysis to: {results_path}\n")


if __name__ == "__main__":
    main()
