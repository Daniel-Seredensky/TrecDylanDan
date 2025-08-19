"""
Search module for the Information Retrieval project.
"""

import asyncio
import aiofiles
import json
from  pathlib import Path
import httpx

import json 
import os

from  pathlib import Path
from functools import partial
from dotenv import load_dotenv; load_dotenv()
from typing import List
from uuid import uuid4

from src.IR_Ensemble.QA_Assistant.rate_limits import gated_cohere_rerank_call
from src.IR_Ensemble.QA_Assistant.daemon_wrapper import JVMDaemon

cohere_client = httpx.AsyncClient(timeout=80.0) 

async def search(queries: List[str], master_query, agentId) -> List[dict]:
    """
    Perform full search pipeline
    """
    results = os.getenv("BM25_RESULTS_PATH")
    path = Path(f"{results}/{agentId}/results-{uuid4()}.jsonl")
    await JVMDaemon.run_bm25_search(queries, path)
    return await rerank_jsonl(path, master_query)

JAVA_CLASSPATH = "src/QA_Assistant/Search/lib/*:."

async def rerank_jsonl(jsonl_path: Path, master_query: str) -> List[dict]:
    """
    Async: Reads a JSONL file at `jsonl_path` (output from your Java Searcher),
    extracts each record's 'segment' text, sends them to Cohere's v2 rerank API
    against `master_query`, and returns a list of the top 75 results
    with only 'title', 'url', 'headings', and 'segmentId'.
    """

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
                "headings":  obj.get("headings"),  
                "segment_id": obj.get("docid")
            })

    # 2) Call Cohere v2 Rerank endpoint
    payload = {
        "model":     "rerank-v3.5",       # or whichever v2 model you prefer
        "query":     master_query,
        "documents": segments,
        "top_n":     75
    }

    cohere_call = partial(                      # bound fn with URL pre‑filled
        cohere_client.post,
        "https://api.cohere.com/v2/rerank"
    )
    resp = await gated_cohere_rerank_call(     
        cohere_call,
        json=payload
    )
    resp.raise_for_status()
    body = resp.json()

    # 3) Sort & select the top 75 by relevance_score
    ranked = sorted(
        body.get("results", []),
        key=lambda x: x["relevance_score"],
        reverse=True
    )[:15] 

    # 4) Build output list with only the requested metadata
    out_list = []
    for r in ranked:
        idx = r["index"]  # v2 returns the original position
        m   = meta[idx]
        out_list.append({
            "title":     m["title"],
            "url":       m["url"],
            "headings":  m["headings"],
            "segment_id": m["segment_id"]
        })
    return out_list

""" async def brave_search(query: str, num_results: int = 3):
    if not os.getenv("BRAVE_API_KEY"):
        raise RuntimeError("BRAVE_API_KEY is not set in the environment.")
    headers = {"Accept": "application/json", "X-Subscription-Token": os.getenv("BRAVE_API_KEY")}
    params = {"q": query, "count": num_results}
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Brave Search API error: {resp.status}")
            data = await resp.json()
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                })
            return results """