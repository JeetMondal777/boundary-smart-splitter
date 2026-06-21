from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HeadingResult:
    level: int
    text: str


class HeadingDetector(ABC):
    """Abstract base class for heading detection."""

    @abstractmethod
    def detect(self, line: str, context: Optional[List[str]] = None) -> Optional[HeadingResult]:
        """Detect if a line is a heading.

        Parameters
        ----------
        line : str
            The line to analyze.
        context : list of str, optional
            Prior lines/context for heuristic detection.

        Returns
        -------
        HeadingResult or None
            A HeadingResult object containing level and clean text if detected, otherwise None.
        """
        ...


class MarkdownHeadingDetector(HeadingDetector):
    """Detects standard Markdown headings (e.g., # Heading, ## Subheading)."""

    _ATX_RE = re.compile(r"^(#{1,6})\s+(.+)$")

    def detect(self, line: str, context: Optional[List[str]] = None) -> Optional[HeadingResult]:
        line = line.strip()
        match = self._ATX_RE.match(line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            return HeadingResult(level=level, text=text)
        return None


class HTMLHeadingDetector(HeadingDetector):
    """Detects HTML headings (e.g., <h1>Title</h1> or role="heading")."""

    _TAG_RE = re.compile(r"^<h([1-6])(?:\s+[^>]*)*>(.*?)</h\1>$", re.IGNORECASE)
    _ARIA_RE = re.compile(
        r"^<[^>]*\brole=[\"']heading[\"'][^>]*\baria-level=[\"']([1-6])[\"'][^>]*>(.*?)</[^>]+>$",
        re.IGNORECASE,
    )

    def detect(self, line: str, context: Optional[List[str]] = None) -> Optional[HeadingResult]:
        line = line.strip()
        
        # 1. Standard tags <h1> to <h6>
        match = self._TAG_RE.match(line)
        if match:
            level = int(match.group(1))
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            return HeadingResult(level=level, text=text)
            
        # 2. ARIA role="heading" with aria-level
        match = self._ARIA_RE.match(line)
        if match:
            level = int(match.group(1))
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            return HeadingResult(level=level, text=text)
            
        return None


class PlainTextHeadingDetector(HeadingDetector):
    """Detects headings in plain text using heuristics:
    - Short lines (e.g. <= 80 characters)
    - Not ending in standard punctuation like period, question mark, comma, etc.
    - All-caps or starting with numbered section pattern
    - Followed by a blank line (if context permits, but here we can just do line heuristics or contextual checks)
    """

    _NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+)$")

    def detect(self, line: str, context: Optional[List[str]] = None) -> Optional[HeadingResult]:
        stripped = line.strip()
        if not stripped:
            return None

        # Ignore very long lines
        if len(stripped) > 80:
            return None

        # Ignore lines ending with common sentence-ending/clause punctuation
        # but allow colons or periods for section numbering
        if stripped[-1] in (".", "?", "!", ",", ";", '"', "'") and not re.match(r"^\d+\.$", stripped):
            if not (re.match(r"^\d+\.$", stripped) or re.match(r"^\s*\d+(?:\.\d+)*\s+[A-Za-z0-9 ]+\.?$", stripped)):
                return None

        # Check for ALL CAPS (with at least one letter)
        is_all_caps = stripped.isupper() and any(c.isalpha() for c in stripped)

        # Check for Numbered Section pattern (e.g. "1.1 Introduction")
        numbered_match = self._NUMBERED_RE.match(stripped)

        if is_all_caps:
            # Let's assign level 1 to ALL CAPS lines
            return HeadingResult(level=1, text=stripped)

        if numbered_match:
            # Determine level based on the number of dots in the section number
            section_num = numbered_match.group(1)
            level = section_num.count(".") + 1
            text = numbered_match.group(2).strip()
            return HeadingResult(level=level, text=text)

        # Short Title Case heuristic
        words = stripped.split()
        if len(words) > 0 and words[0][0].isupper() and len(stripped) < 40:
            return HeadingResult(level=2, text=stripped)

        return None


def select_detector(text: str) -> HeadingDetector:
    """Select the best heading detector based on the content of the text."""
    # Heuristics for detection:
    # 1. HTML headings: <h1 or role="heading"
    if "<h" in text.lower() or "role=\"heading\"" in text.lower() or "role='heading'" in text.lower():
        return HTMLHeadingDetector()
    
    # 2. Markdown headings: any line starting with #, ##, etc.
    # We look for a line starting with # followed by space
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            return MarkdownHeadingDetector()
            
    # 3. Fallback to PlainText
    return PlainTextHeadingDetector()
