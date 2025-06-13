"""
Search module for the Information Retrieval project.
"""

from typing import List
import asyncio
from dotenv import load_dotenv; load_dotenv()
import os
import aiofiles
import httpx
import json
from  pathlib import Path

async def search(queries: List[str], master_query, agentId) -> List[dict]:
    """
    Perform full search pipeline
    """
    results = os.getenv("BM25_RESULTS_PATH")
    path = f"{results}/{agentId}/results.jsonl"  
    await run_bm25_search(queries, path)
    return await rerank_jsonl(path, master_query)

JAVA_CLASSPATH = "src/InfoRetrieval/Search/lib/*:."

async def run_bm25_search(queries: list[str], path: Path) -> None:
    cmd = [
        "java",
        "-cp", JAVA_CLASSPATH,
        "src.InfoRetrieval.Search.Searcher",
        *queries,
        str(path)
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,   # capture if you need it; else use None|DEVNULL
        stderr=asyncio.subprocess.DEVNULL
    )

    try:
        await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("BM25 search timed out after 5 minutes")

    if proc.returncode:
        raise RuntimeError(f"Java search failed [{proc.returncode}]: ")

async def rerank_jsonl(jsonl_path: str, master_query: str) -> str:
    """
    Async: Reads a JSONL file at `jsonl_path` (output from your Java Searcher),
    extracts each record's 'segment' text, sends them to Cohere's v2 rerank API
    against `master_query`, and returns a JSONL string of the top 75 results
    with only 'title', 'url', 'headings', and 'segmentId'.
    """
    # 0) Load your Cohere key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set in environment")

    # 1) Read and buffer all segments + metadata
    segments = []
    meta     = []
    async with aiofiles.open(jsonl_path, mode="r", encoding="utf-8") as f:
        async for line in f:
            obj = json.loads(line)
            segments.append(obj.get("segment", ""))
            meta.append({
                "title":     obj.get("title"),
                "url":       obj.get("url"),
                "headings":  obj.get("headings"),   # sometimes called 'headers'
                "segmentId": obj.get("segmentId")
            })

    # 2) Call Cohere v2 Rerank endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":     "rerank-v3.5",       # or whichever v2 model you prefer
        "query":     master_query,
        "documents": segments,
        "top_n":     75
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.cohere.com/v2/rerank",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        body = resp.json()

    # 3) Sort & select the top 75 by relevance_score
    ranked = sorted(
        body.get("results", []),
        key=lambda x: x["relevance_score"],
        reverse=True
    )[:75]

    # 4) Build output JSONL with only the requested metadata
    out_lines = []
    for r in ranked:
        idx = r["index"]  # v2 returns the original position
        m   = meta[idx]
        out_lines.append(json.dumps({
            "title":     m["title"],
            "url":       m["url"],
            "headings":  m["headings"],
            "segmentId": m["segmentId"]
        }))

    return "\n".join(out_lines)


async def main ():
    queries = ["loving words from the bible", "heart felt messages bible"]
    master = "Looking for loving words from the bible, positive messages that relate to love"
    id = "test"
    print(await search(queries, master, id))

if __name__ == "__main__":
    asyncio.run(main())
