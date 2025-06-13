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
import json
import re

def run_bm25_search(self, queries: list[str], path: str) -> None:
        """
        Helper method to run the Java BM25 search with subprocess
        
        Args:
            queries: List of query strings to search for
        """
        try:
            # Prepare the command
            java_cmd = [
                "java", 
                "-cp", ".:lib/*",  # Adjust classpath as needed
                "src.InfoRetrieval.Search"
            ] + queries + [path]
            
            # Run the Java search
            result = subprocess.run(
                java_cmd,
                capture_output=False,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Java search failed: {result.stderr}")
                
            print("Proctor: BM25 search completed successfully")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("BM25 search timed out after 5 minutes")
        except Exception as e:
            raise RuntimeError(f"Failed to execute BM25 search: {e}")
        

async def get_embedding(client: AsyncOpenAI, text: str, model: str) -> List[float]:
        """Fetch an embedding vector from the OpenAI Embeddings API."""
        response = await client.embeddings.create(
            model= model,
            input=text,
        )
        return response.data[0].embedding

async def reset_index():
    """
    Asynchronously runs AzureJanitor.bash and passes in the absolute path to index_schema.json,
    which is assumed to live next to this .py file.
    """
    # Determine where this .py file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "index_schema.json")
    script_path = os.path.join(base_dir, "BashUtils/AzureJanitor.bash")
    
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Could not find AzureJanitor.bash at {script_path}")
    
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Could not find index_schema.json at {schema_path}")
    
    # Invoke the bash script asynchronously, passing schema_path as the first argument
    proc = await asyncio.create_subprocess_exec(
        "bash", script_path, schema_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        err = stderr.strip() or "Unknown error"
        raise RuntimeError(f"AzureJanitor.bash failed: {err}")
    
    return stdout.strip() if stdout else None

async def _expand_query(query: str) -> List[str]:
    """
    Asynchronously return the synonym-expanded token list for `query`.
    """
    SCRIPT = Path(__file__).with_name("BashUtils/ExpandQuery.sh")  # assumes script is alongside .py
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(SCRIPT), query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err = stderr.strip() or "Unknown error"
            raise RuntimeError(f"ExpandQuery.sh failed with return code {proc.returncode}: {err}")
        
        # Each line is one token
        return [tok for tok in stdout.splitlines() if tok]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find ExpandQuery.sh at {SCRIPT}")
    except Exception as e:
        raise RuntimeError(f"Failed to expand query '{query}': {e}")
        
async def expand_queries(queries: List[str]) -> List[List[str]]:
    """Expand multiple queries concurrently"""
    tasks = [_expand_query(query) for query in queries]
    return await asyncio.gather(*tasks)

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
