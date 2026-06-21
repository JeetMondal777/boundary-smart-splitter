import os
import sys

# Ensure package is on the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boundary_smart_splitter import StructureSplitter, Evaluator

# 1. Define diverse test texts
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
    "Wikipedia Article (Markdown)": WIKIPEDIA_TEXT,
    "Academic Paper (Dirty Text)": ACADEMIC_TEXT,
    "News Article (HTML)": NEWS_TEXT,
    "Python Code with Docstrings": CODE_TEXT
}

def main():
    print("==========================================================")
    print("        RUNNING BOUNDARY-SMART-SPLITTER BENCHMARKS        ")
    print("==========================================================\n")

    # Create fixtures directory and save files for transparency
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Save files to fixtures
    file_mapping = {
        "wikipedia.md": WIKIPEDIA_TEXT,
        "academic.txt": ACADEMIC_TEXT,
        "news.html": NEWS_TEXT,
        "code.py": CODE_TEXT
    }
    for name, content in file_mapping.items():
        with open(os.path.join(fixtures_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    evaluator = Evaluator()
    our_splitter = StructureSplitter(max_chars=1000, min_tokens=100)

    markdown_report = []
    markdown_report.append("# Boundary-Smart-Splitter Benchmark Report\n")
    markdown_report.append("This report evaluates splitters on diverse document types across multiple chunk-quality metrics.\n")

    for doc_name, text in DOCUMENTS.items():
        print(f"Evaluating: {doc_name}...")
        report = evaluator.evaluate(
            text=text,
            splitter=our_splitter,
            compare_with=["recursive", "character"]
        )
        
        doc_header = f"## Document: {doc_name}"
        summary_table = report.summary()
        
        print(doc_header)
        print(summary_table)
        print("\n" + "-"*50 + "\n")
        
        markdown_report.append(doc_header + "\n")
        markdown_report.append(summary_table + "\n")

    # Save benchmark results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_report))

    print(f"Saved benchmark results to: {results_path}")

if __name__ == "__main__":
    main()
