"""HTML -> clean text, and lexical fingerprinting."""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser

_DROP = ("script", "style", "noscript", "svg", "form", "iframe", "figure")


_BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre",
          "td", "dd", "dt"}


def html_to_text(html: str) -> str:
    """Extract readable text, preserving block structure and document order.

    Two things this gets right that the obvious one-liner does not. Taking
    `.text()` over the whole tree with a newline separator splits on every inline
    <b>/<a>, producing one-word lines that wreck n-gram fingerprinting -- so text
    is joined with spaces *within* a block and newlines *between* blocks. And
    `css()` returns nodes grouped by selector rather than in document order,
    which silently scrambles the article, so the tree is walked instead.
    """
    if not html:
        return ""
    tree = HTMLParser(html)
    for tag in _DROP:
        for node in tree.css(tag):
            node.decompose()
    root = tree.body or tree
    blocks: list[str] = []
    for node in root.traverse(include_text=False):
        if node.tag in _BLOCK:
            t = normalize_ws(node.text(separator=" "))
            if t:
                blocks.append(t)
    if not blocks:  # no block markup at all; fall back to the flat text
        return normalize_ws(root.text(separator=" "))
    # traverse() yields ancestors before descendants, so a <td> wrapping a <li>
    # emits the child's text twice. Keep the outer, drop what it already covers.
    out: list[str] = []
    for blk in blocks:
        if not any(blk in kept for kept in out):
            out.append(blk)
    return "\n\n".join(out)


def normalize_ws(text: str) -> str:
    """Collapse whitespace and repair spacing artefacts left by inline markup.

    Stripping inline tags strands spaces before punctuation ("world , this").
    Left alone those shift every word n-gram in the sentence, so verbatim phrase
    reuse would fail to match between two copies of the same text.
    """
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def shingles(text: str, n: int = 5) -> set[str]:
    """Lowercased word n-grams, used to detect verbatim phrase reuse downstream."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def extract_links(html: str) -> list[str]:
    """Outbound hrefs -- the hard, non-semantic evidence of transmission."""
    if not html:
        return []
    out = []
    for node in HTMLParser(html).css("a[href]"):
        # A valueless attribute (`<a href>`) parses to None, and dict.get's
        # default does not apply because the key IS present -- so the fallback
        # has to be `or ""`. Real news pages contain these, and the raw None
        # crashed ingestion mid-run.
        href = node.attributes.get("href") or ""
        if href.startswith("http"):
            out.append(href)
    return out
