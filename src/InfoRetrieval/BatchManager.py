from __future__ import annotations

import asyncio, json, os, time, re
from pathlib import Path
from typing import Dict, List, Any

from azure.search.documents.aio import SearchClient          # async SDK ✔
from azure.core.exceptions import ResourceNotFoundError
from openai import AsyncOpenAI                                # async OpenAI ✔
from src.InfoRetrieval.Utils import get_embedding


class BatchManager:
    """
    Singleton that uploads a JSONL batch to Azure AI Search with client‑side
    embeddings, then waits (non‑blocking) until a sentinel doc is query‑ready.
    """

    _instance: "BatchManager | None" = None

    # --------------------- singleton boilerplate -----------------------------
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, poll_interval: float = 1.0):
        if getattr(self, "_init_complete", False):
            return                                   # already initialised

        self.poll_interval = poll_interval
        self.client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX"),
            credential=os.getenv("SEARCH_ADMIN_KEY"),
        )
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._seen_ids: set[str] = set()              # in‑process dedup cache

        self._init_complete = True

    # --------------------- public API ----------------------------------------
    async def upload_to_azure(self, jsonl_path: str) -> List[Dict]:
        """
        • Parse `jsonl_path`
        • Deduplicate against in‑process cache
        • Embed texts with OpenAI
        • `mergeOrUpload` docs (≤ 1 000)
        • Await sentinel doc’s visibility (non‑blocking)
        """
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(p)

        docs: Dict[str, Dict] = {}
        # ---- 1. load + dedup (sync disk IO is fine) -------------------------
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                doc_id = entry["id"]
                if doc_id in self._seen_ids:
                    continue

                text = entry.get("bestContent", "").strip()
                if not text:
                    continue

                docs[doc_id] = {
                    "id": doc_id,
                    "text": text,
                    "url": entry.get("url", "").strip(),
                    "title": "",
                    "headers": "",
                    # "contentVector" added after embedding
                }
                self._seen_ids.add(doc_id)

        if not docs:
            return []                                 # nothing new

        # ---- 2. embed (async OpenAI) -------------------------------------------------
        texts = [d["text"] for d in docs.values()]
        resp = await self.openai.embeddings.create(model=self.embed_model, input=texts)
        vectors = [e.embedding for e in resp.data]    # order preserved

        for doc, vec in zip(docs.values(), vectors):
            doc["contentVector"] = vec

        # ---- 3. upload (async SDK) ---------------------------------------------------
        await self.client.upload_documents(list(docs.values()))

        # ---- 4. await sentinel -------------------------------------------------------
        sentinel_id = next(iter(docs))
        await self._wait_for_document(sentinel_id)

        return list(docs.values())

    # --------------------- helpers -------------------------------------------
    async def _wait_for_document(self, doc_id: str, max_wait: int = 60):
        """
        Poll GET /docs/{key} every `self.poll_interval` (non‑blocking) until
        the doc is visible or `max_wait` seconds elapse.
        """
        deadline = time.monotonic() + max_wait
        while True:
            try:
                await self.client.get_document(doc_id)      # async call ✔
                return                                      # success
            except ResourceNotFoundError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{doc_id!r} still not visible after {max_wait}s"
                    )
                await asyncio.sleep(self.poll_interval)     # non‑blocking

    # ------------------------------------------------------------------ #
    # Search operations
    # ------------------------------------------------------------------ #

    async def search (self,queries: List[str], seed_paragraph: str) -> Dict[str, Any]:
        """
        Perform a hybrid search by combining keyword search with vector search.
        """
        return await self._perform_hybrid_search(queries, seed_paragraph)
    

    async def _perform_hybrid_search(self, queries: List[str], seed_paragraph: str) -> Dict[str, Any]:
        """Combine keyword search with vector search (semantic configuration ‘default’)."""
        seed_embedding = get_embedding(self.openai,seed_paragraph,self.embed_model)
        combined_query = " ".join(queries)

        search_results = await self.client.search(
            search_text=combined_query,
            vector_queries=[{
                "vector": seed_embedding,
                "k_nearest_neighbors": 75,
                "fields": "contentVector",
            }],
            top=75,
            select=["id", "title", "headers", "url", "text"],
            query_type="semantic",
            semantic_configuration_name="default",
        )

        docs = [{
            "id": r.get("id"),
            "title": r.get("title", ""),
            "headers": r.get("headers", ""),
            "url": r.get("url", ""),
            "text": r.get("text", ""),
            "search_score": r.get("@search.score", 0),
            "search_reranker_score": r.get("@search.reranker_score", 0),
        } for r in search_results]

        return {
            "total_results": len(docs),
            "documents": docs,
            "queries_used": queries,
            "seed_paragraph": seed_paragraph,
        }

