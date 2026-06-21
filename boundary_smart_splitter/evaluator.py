from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

from .heading_detector import select_detector

class EvaluationReport:
    """Represents the compiled metrics comparison report."""

    def __init__(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.data = data

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return raw metrics dictionary."""
        return self.data

    def summary(self) -> str:
        """Return a formatted Markdown table comparing metrics of all splitters."""
        if not self.data:
            return "No evaluation data."

        splitters = list(self.data.keys())
        # Prioritize ours or "smart" splitter to be first in the column list
        ours_key = None
        for k in splitters:
            if "smart" in k.lower() or "ours" in k.lower() or "structure" in k.lower():
                ours_key = k
                break
        if ours_key:
            splitters.remove(ours_key)
            splitters.insert(0, ours_key)

        metrics = [
            ("avg_tokens_per_chunk", "Avg tokens/chunk", "{:.1f}"),
            ("token_distribution_cv", "CV (consistency)", "{:.2f}"),
            ("pct_chunks_under_100_tokens", "Chunks < 100 tokens", "{:.1%}"),
            ("pct_chunks_over_512_tokens", "Chunks > 512 tokens", "{:.1%}"),
            ("cross_topic_mixing_rate", "Cross-topic mixing", "{:.1%}"),
            ("heading_preservation_rate", "Heading preservation", "{:.1%}"),
            ("processing_time_ms", "Processing time", "{:.1f}ms"),
        ]

        header = "| Metric | " + " | ".join(splitters) + " |"
        divider = "|:---| " + " | ".join([":---" for _ in splitters]) + " |"
        rows = []
        for metric_key, metric_name, fmt in metrics:
            row_parts = [metric_name]
            for s in splitters:
                val = self.data[s].get(metric_key, None)
                if val is None:
                    row_parts.append("N/A")
                else:
                    row_parts.append(fmt.format(val))
            rows.append("| " + " | ".join(row_parts) + " |")

        return "\n".join([header, divider] + rows)


class Evaluator:
    """Utility to evaluate and compare text splitters on chunk quality metrics."""

    def __init__(self, length_function: Union[str, Callable[[str], int]] = "cl100k_base") -> None:
        self.length_function = length_function
        
        # Resolve length function
        if isinstance(length_function, str):
            try:
                import tiktoken
                try:
                    enc = tiktoken.get_encoding(length_function)
                    self._len_fn = lambda text: len(enc.encode(text))
                except (ValueError, KeyError):
                    try:
                        enc = tiktoken.encoding_for_model(length_function)
                        self._len_fn = lambda text: len(enc.encode(text))
                    except (ValueError, KeyError):
                        self._len_fn = len
            except ImportError:
                self._len_fn = len
        elif callable(length_function):
            self._len_fn = length_function
        else:
            self._len_fn = len

    def evaluate(
        self,
        text: str,
        splitter: Any,
        compare_with: Optional[List[Union[str, Any]]] = None
    ) -> EvaluationReport:
        """Evaluate a text splitter and optionally compare with other splitters.

        Parameters
        ----------
        text : str
            The input document to split.
        splitter : Any
            The main splitter to evaluate (e.g. StructureSplitter).
        compare_with : list of (str or splitter instance), optional
            Competitor splitters to run against. Can be keys like "recursive",
            "character", "nltk", or pre-configured instances.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # 1. Run main splitter
        main_name = type(splitter).__name__
        results[main_name] = self._run_and_evaluate(text, splitter)

        # 2. Run competitors
        if compare_with:
            for comp in compare_with:
                if isinstance(comp, str):
                    comp_inst = self._instantiate_competitor(comp, splitter)
                    if comp_inst is not None:
                        comp_name = f"{type(comp_inst).__name__} ({comp})"
                        results[comp_name] = self._run_and_evaluate(text, comp_inst)
                else:
                    comp_name = type(comp).__name__
                    results[comp_name] = self._run_and_evaluate(text, comp)

        return EvaluationReport(results)

    def _instantiate_competitor(self, key: str, main_splitter: Any) -> Optional[Any]:
        # Determine standard limits based on main splitter
        max_chars = getattr(main_splitter, "max_chars", 1500)
        if max_chars is None:
            max_chars = 1500

        if key == "recursive":
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                return RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=0)
            except ImportError:
                warnings.warn("langchain-text-splitters not installed, skipping recursive splitter.", ImportWarning)
                return None
        elif key == "character":
            try:
                from langchain_text_splitters import CharacterTextSplitter
                return CharacterTextSplitter(chunk_size=max_chars, chunk_overlap=0, separator="\n\n")
            except ImportError:
                warnings.warn("langchain-text-splitters not installed, skipping character splitter.", ImportWarning)
                return None
        elif key == "nltk":
            try:
                from langchain_text_splitters import NLTKTextSplitter
                return NLTKTextSplitter(chunk_size=max_chars, chunk_overlap=0)
            except ImportError:
                warnings.warn("langchain-text-splitters not installed, skipping NLTK splitter.", ImportWarning)
                return None
        else:
            raise ValueError(f"Unknown competitor key: {key!r}")

    def _run_and_evaluate(self, text: str, splitter: Any) -> Dict[str, Any]:
        # Measure processing time
        start_time = time.perf_counter()
        
        # Split text
        if hasattr(splitter, "split_text"):
            chunks = splitter.split_text(text)
        elif hasattr(splitter, "split"):
            chunks = splitter.split(text)
        elif callable(splitter):
            chunks = splitter(text)
        else:
            raise TypeError(f"Splitter {splitter!r} does not have a split method and is not callable")
            
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000.0

        if not chunks:
            return {
                "avg_tokens_per_chunk": 0.0,
                "token_distribution_cv": 0.0,
                "pct_chunks_under_100_tokens": 0.0,
                "pct_chunks_over_512_tokens": 0.0,
                "cross_topic_mixing_rate": 0.0,
                "heading_preservation_rate": 1.0,
                "processing_time_ms": processing_time_ms,
            }

        # Calculate token counts
        token_counts = [self._len_fn(c) for c in chunks]
        avg_tokens = statistics.mean(token_counts)
        
        if len(token_counts) > 1:
            stdev = statistics.stdev(token_counts)
            cv = stdev / avg_tokens if avg_tokens > 0 else 0.0
        else:
            cv = 0.0

        pct_under_100 = sum(1 for tc in token_counts if tc < 100) / len(token_counts)
        pct_over_512 = sum(1 for tc in token_counts if tc > 512) / len(token_counts)

        # Heading detection and mapping for structure metrics
        detector = select_detector(text)
        headings = []
        for line in text.split("\n"):
            line_stripped = line.strip()
            heading_res = detector.detect(line_stripped)
            if heading_res is not None:
                headings.append(heading_res)

        # Heading Preservation Rate
        # A heading is preserved if it starts any of the chunks (after stripping)
        chunk_start_headings = set()
        for c in chunks:
            first_line = c.split("\n")[0].strip()
            res = detector.detect(first_line)
            if res is not None:
                chunk_start_headings.add(res.text)

        preserved_count = sum(1 for h in headings if h.text in chunk_start_headings)
        heading_preservation = preserved_count / len(headings) if headings else 1.0

        # Cross-Topic Mixing Rate
        # A chunk is mixed if it contains any heading in its body, but NOT at the start
        mixed_count = 0
        for c in chunks:
            lines = c.split("\n")
            if len(lines) <= 1:
                continue
            # Check subsequent lines for heading detection
            for line in lines[1:]:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                res = detector.detect(line_stripped)
                if res is not None:
                    mixed_count += 1
                    break

        cross_topic_mixing = mixed_count / len(chunks)

        return {
            "avg_tokens_per_chunk": avg_tokens,
            "token_distribution_cv": cv,
            "pct_chunks_under_100_tokens": pct_under_100,
            "pct_chunks_over_512_tokens": pct_over_512,
            "cross_topic_mixing_rate": cross_topic_mixing,
            "heading_preservation_rate": heading_preservation,
            "processing_time_ms": processing_time_ms,
        }
