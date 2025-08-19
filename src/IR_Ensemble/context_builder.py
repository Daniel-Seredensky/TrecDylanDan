from __future__ import annotations

"""ContextProctor – queue‑driven orchestrator

• Maintains a **max of three concurrent workers** (configurable).
• Each worker processes one *batch* of 3‑5 questions produced from the input
  list and submits them to `assess_questions`.
• The **first three workers are staggered by 3 s** so their PLAN/TOOL/UPDATE
  phases stay out of sync, avoiding spikes on stage‑specific buckets such as
  the Cohere 10 req/min limiter.
• Subsequent workers start immediately when a slot frees up, which preserves
  the natural phase offset.

Environment
-----------
`CONTEXT_PATH` must be defined in your `.env`.
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ── third‑party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import aiofiles

# ── internal ────────────────────────────────────────────────────────────────
from src.IR_Ensemble.QA_Assistant.question_eval import assess_questions


class ContextProctor:
    """Runs `assess_questions` with **queue‑managed** concurrency."""

    MAX_WORKERS: int = 5          # how many workers run in parallel 
    STAGGER_SEC: float = 1.0      # delay between first, second, third starts
    BATCH_SIZE: int = 2           # questions per worker‑batch (‑5 recommended)

    # ────────────────────────────── init ────────────────────────────────
    def __init__(self, client: AsyncAzureOpenAI, questions: List[Dict[str, str]],num: int ):
        load_dotenv()
        self.client = client
        self.questions = questions
        self.num = num

        # Slice into (index, batch) tuples so we can emit results in order
        self._batches: List[Tuple[int, List[Dict[str, str]]]] = [
            (idx, questions[i : i + self.BATCH_SIZE])
            for idx, i in enumerate(range(0, len(questions), self.BATCH_SIZE))
        ]

        self._results: List[str | None] = [None] * len(self._batches)
        self._queue: asyncio.Queue[Tuple[int, List[Dict[str, str]]]] = asyncio.Queue()
        self.context_path = Path(f"{os.getenv('CONTEXT_PATH')}{self.num}.txt")
        for item in self._batches:
            self._queue.put_nowait(item)

    # ───────────────────────── public API ───────────────────────────────
    async def create_context(self) -> None:
        """Main entry – dispatch the worker pool, write `CONTEXT_PATH`."""

        # Launch workers
        workers = [
            asyncio.create_task(self._worker(i), name=f"CTX‑worker‑{i}")
            for i in range(self.MAX_WORKERS)
        ]

        # Wait until every batch has been handled
        await self._queue.join()

        # Cancel idle workers (they exit quietly)
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Concatenate results in original batch order
        sep = "\n===================================\n"
        total_context = sep.join(filter(None, self._results))

        async with aiofiles.open(self.context_path, "a") as f:
            await f.write(total_context)

    # ─────────────────────────── worker loop ────────────────────────────
    async def _worker(self, worker_idx: int):
        """A single worker task that pulls batches off the queue."""
        # Stagger only the first launch of each worker
        if worker_idx:
            await asyncio.sleep(worker_idx * self.STAGGER_SEC)

        while True:
            try:
                batch_idx, batch = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # nothing left → exit (will be cancelled by caller)

            try:
                # One batch ➜ many questions ➜ gather
                result_str = await self._process_batch(batch)
                self._results[batch_idx] = result_str
            finally:
                self._queue.task_done()

    # ───────────────────────── batch helper ─────────────────────────────
    async def _process_batch(self, batch: List[Dict[str, str]]) -> str:
        """Call `assess_questions` for each question in the batch concurrently."""
        # Convert question dict ➜ JSON string (per original code expectations)
        stringified = [json.dumps(q) for q in batch]
        contexts = await assess_questions("\n".join(stringified),self.client,self.num)
        return json.dumps(contexts)
    

