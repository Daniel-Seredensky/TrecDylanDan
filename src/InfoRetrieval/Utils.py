"""
    Uility functions for the project
    Some are wrappers for curl scripts, some are wrappers for OpenAI calls
"""

import subprocess
import os
from openai import AsyncOpenAI
from typing import List
from pathlib import Path
import asyncio
import lucene
from java.nio.file import Paths
from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, TermQuery


# fetch full document text from index
async def get_document_text(doc_id: str, index_dir: str) -> str:
    directory = NIOFSDirectory.open(Paths.get(str(Path(index_dir).resolve())))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    query = TermQuery(Term("id", doc_id))
    top = searcher.search(query, 1)
    if top.totalHits.value == 0:
        reader.close()
        directory.close()
        raise KeyError(f"Document ID '{doc_id}' not found in {index_dir}")
    doc = searcher.doc(top.scoreDocs[0].doc)
    text = doc.get("content")
    reader.close()
    directory.close()
    return text or ""

# assemble documents for LLM
async def document_selection(
    document_ids: List[str],
    bestFragment: bool,
    *,
    index_dir: str = "DerivedData/CluewebIndex",
    fragment_char_limit: int = 1000,
) -> str:
    if not document_ids:
        raise ValueError("document_ids must contain at least one ID")
    doc_ids = document_ids[:3]

    # choose longest paragraph
    def _fragment(txt: str) -> str:
        parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
        if not parts:
            return txt[:fragment_char_limit]
        long_p = max(parts, key=len)
        if len(long_p) <= fragment_char_limit:
            return long_p
        cut = long_p.rfind(" ", 0, fragment_char_limit)
        return long_p[: cut if cut != -1 else fragment_char_limit] + "…"

    # concurrent reads
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(None, get_document_text, doc_id, index_dir)
        for doc_id in doc_ids
    ]
    texts = await asyncio.gather(*tasks)

    # optional trimming
    texts = [_fragment(t) if bestFragment else t for t in texts]

    # markdown join
    blocks = [f"### {did}\n{txt}" for did, txt in zip(doc_ids, texts)]
    return "\n\n".join(blocks)