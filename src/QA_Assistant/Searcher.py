"""
Search module for the Information Retrieval project.
"""

import asyncio
import aiofiles
import json
from  pathlib import Path
import aiohttp
import httpx
import httpx

import json 
import os

from  pathlib import Path
from functools import partial
from dotenv import load_dotenv; load_dotenv()
from typing import List
from uuid import uuid4

from src.QA_Assistant.rate_limits import gated_cohere_rerank_call



async def search(queries: List[str], master_query, agentId) -> List[dict]:
    """
    Perform full search pipeline
    """
    results = os.getenv("BM25_RESULTS_PATH")
    path = Path(f"{results}/{agentId}/results-{uuid4()}.jsonl")
    await run_bm25_search(queries, path)
    return await rerank_jsonl(path, master_query)

JAVA_CLASSPATH = "src/QA_Assistant/Search/lib/*:."

async def run_bm25_search(queries: list[str], path: Path) -> None:
    cmd = [
        "java",
        "-cp", JAVA_CLASSPATH,
        "src.QA_Assistant.Search.Searcher",
        *queries,
        str(path)
    ]
    print(cmd)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,   # capture output for debugging
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy()  # Pass current environment, including .env vars
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("BM25 search timed out after 5 minutes")

    if proc.returncode:
        print("Java process stdout:\n", stdout.decode())
        print("Java process stderr:\n", stderr.decode())
        raise RuntimeError(f"Java search failed [{proc.returncode}]: ")

async def rerank_jsonl(jsonl_path: Path, master_query: str) -> List[dict]:
    """
    Async: Reads a JSONL file at `jsonl_path` (output from your Java Searcher),
    extracts each record's 'segment' text, sends them to Cohere's v2 rerank API
    against `master_query`, and returns a list of the top 75 results
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
                "headings":  obj.get("headings"),  
                "segment_id": obj.get("docid")
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

    async with httpx.AsyncClient(timeout=80.0) as client:
        cohere_call = partial(                      # bound fn with URL pre‑filled
            client.post,
            "https://api.cohere.com/v2/rerank"
        )
        resp = await gated_cohere_rerank_call(     
            cohere_call,
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
    )[:5] # 15 right now for testing

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

async def brave_search(query: str, num_results: int = 3):
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
            return results

async def main ():
    queries = ["loving words from the bible", "heart felt messages bible"]
    master = "Looking for loving words from the bible, positive messages that relate to love"
    id = "test"
    print(await search(queries, master, id))
    
    # Test Brave Search
    """ print("\n--- Testing Brave Search ---")
    try:
        brave_results = await brave_search("TREC DRAGUN track", 3)
        print(f"Found {len(brave_results)} Brave search results:")
        for i, result in enumerate(brave_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Description: {result['description'][:100]}...")
            print()
    except Exception as e:
        print(f"Brave Search test failed: {e}") """

async def test_brave_search():
    """Test function specifically for Brave Search API"""
    test_queries = [
        "TREC DRAGUN track",
        "information retrieval evaluation",
        "credibility analysis pipeline"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        try:
            results = await brave_search(query, 2)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Description: {result['description'][:80]}...")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
    # Uncomment the line below to run the dedicated Brave Search test
    # asyncio.run(test_brave_search())
