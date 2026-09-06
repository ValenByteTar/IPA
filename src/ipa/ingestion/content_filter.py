"""Content filter â€” site-agnostic content extraction from HTML.

Inspired by Crawl4AI's RelevantContentFilter, this module extracts the
main content from any web page without per-site configuration.

Strategy:
  1. Parse HTML with BeautifulSoup
  2. Remove excluded tags (nav, footer, header, script, style, ...)
  3. Remove elements whose class/id match negative patterns (sidebar, ads, ...)
  4. Extract text blocks from remaining elements
  5. Score each block with BM25 against the page's title/h1/meta description
  6. Return the top-scoring blocks as clean text

The BM25 scoring ensures that the content most relevant to the page's
topic rises to the top, while boilerplate (cookie notices, related posts,
social share buttons) sinks.

Usage:
  from ipa.ingestion.content_filter import ContentFilter

  cf = ContentFilter()
  text = cf.extract(html_string)
  # Or with a query to bias toward specific content:
  text = cf.extract(html_string, query="adversarial attacks")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Tags that are typically content
_INCLUDED_TAGS: set[str] = {
    # Primary structure
    "article", "main", "section", "div",
    # Text content
    "p", "span", "blockquote", "pre", "code",
    # Headers
    "h1", "h2", "h3", "h4", "h5", "h6",
    # Lists
    "ul", "ol", "li", "dl", "dt", "dd",
    # Tables
    "table", "thead", "tbody", "tr", "td", "th",
    # Other semantic elements
    "figure", "figcaption", "details", "summary",
    # Text formatting
    "em", "strong", "b", "i", "mark", "small",
    # Rich content
    "time", "address", "cite", "q",
}

# Tags that are never content â€” removed entirely
_EXCLUDED_TAGS: set[str] = {
    "nav", "footer", "header", "aside",
    "script", "style", "form", "iframe", "noscript",
    "svg", "canvas", "template",
}

# Patterns in class/id that indicate non-content elements.
# These are matched as whole words (word-boundary), not substrings,
# to avoid false positives like "article-header" matching "header".
_NEGATIVE_PATTERNS = re.compile(
    r"\b(?:nav(?:igation)?|footer|sidebar|ads|advert|promo|social|share|"
    r"comment|cookie|banner|popup|modal|overlay|widget|related|"
    r"newsletter|subscribe|signup|login|breadcrumb|pagination|"
    r"menu|toolbar|skip-link|screen-reader)\b",
    re.I,
)

# Patterns that strongly indicate main content
_POSITIVE_PATTERNS = re.compile(
    r"\b(?:article|content|main|body|post|entry|story|text|prose)\b",
    re.I,
)

# Tags that are protected from negative pattern removal.
# Even if their class/id matches a negative pattern, these tags
# are kept because they are semantically content-bearing.
_PROTECTED_TAGS: set[str] = {
    "article", "main", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "code", "blockquote", "figcaption", "time", "address",
    "cite", "q", "em", "strong", "b", "i", "mark", "small",
}

# Minimum word count for a block to be considered content
_MIN_WORD_COUNT = 5

# Minimum text length (chars) for a block to be scored
_MIN_CHAR_COUNT = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """A candidate content block extracted from HTML."""
    text: str
    tag: str
    score: float = 0.0
    word_count: int = 0
    char_count: int = 0
    is_header: bool = False
    # Positive/negative signals from class/id
    positive_signals: int = 0
    negative_signals: int = 0


@dataclass
class ExtractionResult:
    """Result of content extraction."""
    text: str
    title: str
    blocks_used: int
    blocks_total: int
    query_used: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ContentFilter
# ---------------------------------------------------------------------------

class ContentFilter:
    """Site-agnostic content extraction â€” no per-site config needed.

    Extracts main content from any HTML page using:
    - Tag-based filtering (remove nav, footer, script, etc.)
    - Class/id pattern matching (remove sidebar, ads, social, etc.)
    - BM25 scoring of remaining blocks against page topic
    - Text density heuristics

    The BM25 scoring uses rank_bm25 (already a project dependency) to
    rank text blocks by relevance to the page's title, h1, and meta
    description. This ensures the main article content rises above
    boilerplate without knowing the site's CSS structure.
    """

    def __init__(
        self,
        min_word_count: int = _MIN_WORD_COUNT,
        min_char_count: int = _MIN_CHAR_COUNT,
        max_blocks: int = 50,
    ) -> None:
        self.min_word_count = min_word_count
        self.min_char_count = min_char_count
        self.max_blocks = max_blocks

    def extract(
        self,
        html: str,
        query: str | None = None,
        base_url: str = "",
    ) -> str:
        """Extract main content text from HTML.

        Args:
            html: Raw HTML string.
            query: Optional query to bias content selection. If None,
                uses the page's title + h1 + meta description as the query.
            base_url: Base URL for resolving relative links (unused in
                text extraction but available for future link extraction).

        Returns:
            Clean text of the main content.
        """
        result = self.extract_detailed(html, query=query, base_url=base_url)
        return result.text

    def extract_detailed(
        self,
        html: str,
        query: str | None = None,
        base_url: str = "",
    ) -> ExtractionResult:
        """Extract main content with detailed metadata.

        Args:
            html: Raw HTML string.
            query: Optional query to bias content selection.
            base_url: Base URL for future link resolution.

        Returns:
            ExtractionResult with text, title, block counts, and metadata.
        """
        if not html or not html.strip():
            return ExtractionResult(text="", title="", blocks_used=0, blocks_total=0)

        soup = BeautifulSoup(html, "lxml")

        # Extract page metadata for BM25 query
        title = self._extract_title(soup)
        meta_desc = self._extract_meta(soup, "description")
        meta_keywords = self._extract_meta(soup, "keywords")
        h1_text = self._extract_h1(soup)

        # Build the BM25 query: title + h1 + meta description + keywords
        # If user provides a query, prepend it (higher weight)
        page_topic = " ".join(filter(None, [
            title, h1_text, meta_desc, meta_keywords,
        ]))

        bm25_query = query if query else page_topic
        if query and page_topic:
            # Combine user query with page topic for better scoring
            bm25_query = f"{query} {page_topic}"

        # Remove excluded tags entirely
        self._remove_excluded_tags(soup)

        # Remove elements matching negative patterns
        self._remove_negative_pattern_elements(soup)

        # Extract candidate text blocks
        blocks = self._extract_text_blocks(soup)

        if not blocks:
            return ExtractionResult(
                text="", title=title, blocks_used=0, blocks_total=0,
                query_used=bm25_query,
                metadata={"description": meta_desc, "keywords": meta_keywords},
            )

        # Score blocks with BM25
        self._score_blocks_bm25(blocks, bm25_query)

        # Adjust scores with heuristics (headers, positive/negative signals)
        self._adjust_scores_heuristics(blocks)

        # Select top blocks (preserving document order)
        selected = self._select_blocks(blocks)

        # Build final text
        text = "\n\n".join(b.text for b in selected)

        return ExtractionResult(
            text=text,
            title=title,
            blocks_used=len(selected),
            blocks_total=len(blocks),
            query_used=bm25_query,
            metadata={"description": meta_desc, "keywords": meta_keywords},
        )

    def extract_markdown(
        self,
        html: str,
        query: str | None = None,
        base_url: str = "",
    ) -> str:
        """Extract content as simple markdown (headers preserved).

        Similar to extract() but preserves heading structure (#, ##, ###)
        for better downstream chunking.
        """
        if not html or not html.strip():
            return ""

        soup = BeautifulSoup(html, "lxml")

        # Remove excluded and negative pattern elements
        self._remove_excluded_tags(soup)
        self._remove_negative_pattern_elements(soup)

        # Extract blocks with tag info for markdown formatting
        blocks = self._extract_text_blocks(soup)
        if not blocks:
            return ""

        # Score and select
        title = self._extract_title(soup)
        h1_text = self._extract_h1(soup)
        meta_desc = self._extract_meta(soup, "description")
        page_topic = " ".join(filter(None, [title, h1_text, meta_desc]))
        bm25_query = query if query else page_topic
        if query and page_topic:
            bm25_query = f"{query} {page_topic}"

        self._score_blocks_bm25(blocks, bm25_query)
        self._adjust_scores_heuristics(blocks)
        selected = self._select_blocks(blocks)

        # Build markdown with header levels
        parts: list[str] = []
        if title:
            parts.append(f"# {title}\n")

        for block in selected:
            if block.is_header:
                level = self._header_level(block.tag)
                prefix = "#" * level
                parts.append(f"{prefix} {block.text}\n")
            else:
                parts.append(block.text)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return ""

    def _extract_meta(self, soup: BeautifulSoup, name: str) -> str:
        """Extract meta tag content by name."""
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return meta["content"].strip()
        # Also check property attribute (OpenGraph)
        meta = soup.find("meta", attrs={"property": f"og:{name}"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        return ""

    def _extract_h1(self, soup: BeautifulSoup) -> str:
        """Extract first h1 text."""
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _remove_excluded_tags(self, soup: BeautifulSoup) -> None:
        """Remove all excluded tags from the soup."""
        for tag_name in _EXCLUDED_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Also remove HTML comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

    def _remove_negative_pattern_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements whose class or id matches negative patterns.

        Protected tags (article, p, h1-h6, pre, etc.) are never removed
        even if their class matches a negative pattern, because they are
        semantically content-bearing. Structural tags (body, html, head)
        are also never removed.

        Additionally, any element that contains protected tags as
        descendants is not removed, even if its class matches a negative
        pattern. This prevents false positives like "flex-wrap-footer"
        from removing the entire page content.
        """
        # Tags that should never be removed by pattern matching
        never_remove = _PROTECTED_TAGS | {"body", "html", "head", "title"}

        for element in soup.find_all(True):
            # Protect content-bearing and structural tags from removal
            if element.name in never_remove:
                continue

            # attrs can be None in some BeautifulSoup versions
            attrs = element.attrs if element.attrs is not None else {}
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_str = " ".join(classes) if classes else ""

            # Check id
            elem_id = attrs.get("id", "") or ""

            # Check role attribute (aria roles)
            role = attrs.get("role", "") or ""

            combined = f"{class_str} {elem_id} {role}"

            if _NEGATIVE_PATTERNS.search(combined):
                # Check if this element contains protected tags as descendants.
                # If it does, don't remove it â€” the content inside is valuable.
                has_protected_descendant = element.find(_PROTECTED_TAGS) is not None
                if has_protected_descendant:
                    continue
                element.decompose()
                continue

            # Check for aria-hidden="true"
            aria_hidden = attrs.get("aria-hidden", "") or ""
            if aria_hidden.lower() == "true":
                element.decompose()

    def _extract_text_blocks(self, soup: BeautifulSoup) -> list[TextBlock]:
        """Extract candidate text blocks from the parsed HTML.

        Walks the DOM tree and collects text from content-bearing elements.
        """
        blocks: list[TextBlock] = []
        body = soup.find("body") or soup

        def _has_positive_signal(tag: Tag) -> int:
            attrs = tag.attrs if tag.attrs is not None else {}
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_str = " ".join(classes) if classes else ""
            class_str += " " + (attrs.get("id", "") or "")
            return len(_POSITIVE_PATTERNS.findall(class_str))

        def _has_negative_signal(tag: Tag) -> int:
            attrs = tag.attrs if tag.attrs is not None else {}
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_str = " ".join(classes) if classes else ""
            class_str += " " + (attrs.get("id", "") or "")
            return len(_NEGATIVE_PATTERNS.findall(class_str))

        # Find all content-bearing elements with direct text
        for element in body.find_all(_INCLUDED_TAGS):
            # Skip if inside an already-processed parent
            # (we want leaf-most content blocks)
            has_child_content = any(
                child.name in _INCLUDED_TAGS
                for child in element.children
                if isinstance(child, Tag)
            )
            if has_child_content and element.name not in (
                "h1", "h2", "h3", "h4", "h5", "h6",
                "pre", "code", "blockquote",
            ):
                continue

            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            word_count = len(text.split())
            char_count = len(text)

            if word_count < self.min_word_count and char_count < self.min_char_count:
                continue

            is_header = element.name in ("h1", "h2", "h3", "h4", "h5", "h6")

            block = TextBlock(
                text=text,
                tag=element.name or "div",
                word_count=word_count,
                char_count=char_count,
                is_header=is_header,
                positive_signals=_has_positive_signal(element),
                negative_signals=_has_negative_signal(element),
            )
            blocks.append(block)

        return blocks

    def _score_blocks_bm25(
        self,
        blocks: list[TextBlock],
        query: str,
    ) -> None:
        """Score blocks using BM25 against the query.

        Uses rank_bm25.BM25Okapi (already a project dependency).
        The query is typically the page title + h1 + meta description.
        """
        if not blocks or not query:
            # No query â€” give all blocks equal score
            for b in blocks:
                b.score = 1.0
            return

        # Tokenize: simple lowercase split (rank_bm25 doesn't include a tokenizer)
        query_tokens = self._tokenize(query)

        # Corpus: each block's text tokenized
        corpus_tokens = [self._tokenize(b.text) for b in blocks]

        if not any(corpus_tokens):
            for b in blocks:
                b.score = 1.0
            return

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)

        for block, score in zip(blocks, scores):
            block.score = float(score)

    def _adjust_scores_heuristics(self, blocks: list[TextBlock]) -> None:
        """Apply heuristic adjustments to BM25 scores.

        - Headers get a boost (they define structure)
        - Blocks with positive class/id signals get a boost
        - Blocks with negative signals get penalized
        - Very short blocks get penalized
        - Very long blocks get a small boost (likely main content)
        """
        for block in blocks:
            # Header boost
            if block.is_header:
                block.score *= 1.5

            # Positive signal boost
            if block.positive_signals > 0:
                block.score *= (1.0 + 0.1 * block.positive_signals)

            # Negative signal penalty
            if block.negative_signals > 0:
                block.score *= max(0.1, 1.0 - 0.2 * block.negative_signals)

            # Length-based adjustments
            if block.word_count < 10:
                block.score *= 0.5  # Very short â€” likely boilerplate
            elif block.word_count > 100:
                block.score *= 1.2  # Long â€” likely main content

            # Ensure non-negative
            block.score = max(0.0, block.score)

    def _select_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """Select the top-scoring blocks, preserving document order.

        Strategy:
        - Always include headers (they provide structure)
        - Include top-scoring content blocks
        - Limit to max_blocks to avoid excessive output
        """
        if not blocks:
            return []

        # Sort by score descending
        sorted_blocks = sorted(blocks, key=lambda b: b.score, reverse=True)

        # Select top N
        top_set = set()
        threshold_score = 0.0

        # Always include headers
        for b in blocks:
            if b.is_header:
                top_set.add(id(b))

        # Add top-scoring content blocks
        content_blocks = [b for b in sorted_blocks if not b.is_header]
        n_content = min(self.max_blocks - len(top_set), len(content_blocks))

        if n_content > 0:
            # Use a score threshold: include blocks with score > 0
            # and at least some fraction of the max score
            if content_blocks:
                max_score = max(b.score for b in content_blocks)
                threshold = max_score * 0.05 if max_score > 0 else 0.0

                for b in content_blocks[:n_content]:
                    if b.score >= threshold:
                        top_set.add(id(b))

        # Return in document order
        return [b for b in blocks if id(b) in top_set]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25.

        Lowercase, split on non-alphanumeric, filter empty.
        """
        return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]

    @staticmethod
    def _header_level(tag: str) -> int:
        """Get heading level from tag name (h1=1, h2=2, ...)."""
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return int(tag[1])
        return 1

