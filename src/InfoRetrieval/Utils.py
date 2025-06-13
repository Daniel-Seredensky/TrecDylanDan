"""
    Uility functions for the project
    Some are wrappers for curl scripts, some are wrappers for OpenAI calls
"""

from typing import List
from pathlib import Path
import asyncio
import json
import re


def _extract_text_field(obj: dict) -> str:
    """
    Return the first field that plausibly stores the raw text of a segment.
    Extend the list below if your corpus uses another key.
    """
    for key in ("segment_text", "contents", "text", "Clean-Text", "body"):
        if key in obj and obj[key]:
            return obj[key]
    return ""

def get_document_text(doc_or_segment_id: str, jsonl_path: str) -> str:
    """
    Assemble the *full* document text by concatenating every segment that belongs
    to the document designated by `doc_or_segment_id`.

    Args:
        doc_or_segment_id: Either a full segment_id **or** the plain doc-id
                           (everything before the '#').
        jsonl_path:        Path to the corpus .jsonl file.

    Returns:
        The combined text of all segments, ordered by their numeric index.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    output = ""

    lines = open(path, "r").readlines()
    for line in lines:
        data = json.loads(line)
        segment_id = data["segment_id"]
        if str.startswith(segment_id, doc_or_segment_id):
            output += data["segment"]

    return output



async def document_selection(
    document_ids: List[str],
    bestFragment: bool,
    jsonl_path: str,
    fragment_char_limit: int = 1000
) -> str:
    """
    For up to the first three *unique* documents referenced by
    `document_ids`, fetch either the full text (default) or the
    “best” paragraph-sized fragment.

    Args:
        document_ids:      List of segment-ids or doc-ids.
        bestFragment:      If True, return only the best fragment.
        jsonl_path:        Path to the corpus .jsonl file.
        fragment_char_limit: Character cap for fragments.

    Returns:
        Markdown-formatted string with one block per document.
    """
    if not document_ids:
        raise ValueError("document_ids must contain at least one ID")

    # Collapse to the first three unique doc-ids, preserving order
    seen, doc_queue = set(), []
    for raw_id in document_ids:
        doc_id = raw_id.split("#", 1)[0]
        if doc_id not in seen:
            seen.add(doc_id)
            doc_queue.append(doc_id)
        if len(doc_queue) == 3:
            break

    def _best_fragment(text: str) -> str:
        """Pick the longest paragraph ≤ fragment_char_limit, trimming if needed."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:                       # empty doc
            return text[:fragment_char_limit]

        best = max(paragraphs, key=len)
        if len(best) <= fragment_char_limit:
            return best

        cut = best.rfind(" ", 0, fragment_char_limit)
        return best[: cut if cut != -1 else fragment_char_limit] + "…"

    loop = asyncio.get_running_loop()
    # fan-out fetches to a default thread pool
    tasks = [
        loop.run_in_executor(None, get_document_text, doc_id, jsonl_path)
        for doc_id in doc_queue
    ]
    docs_full = await asyncio.gather(*tasks)

    docs_final = [
        _best_fragment(txt) if bestFragment else txt
        for txt in docs_full
    ]

    # assemble markdown blocks
    blocks = [
        f"### {doc_id}\n{doc_text}"
        for doc_id, doc_text in zip(doc_queue, docs_final)
    ]
    return "\n\n".join(blocks)
