from __future__ import annotations

"""
Bucket throughput monitor
=========================
Polls the three `AsyncTokenBucket` instances every *0.5* seconds and appends
current **remaining capacity** ( ≡ "tokens/requests still available in this 60‑s window")
to a CSV file so you can graph usage live.

CSV columns
-----------
```
time_iso,Cohere_req,OpenAI_req,OpenAI_tok
```

Usage
-----
```python
from rate_limits import (
    openai_req_limiter,
    openai_tok_limiter,
    cohere_rerank_limiter,
)
from bucket_monitor import BucketMonitor

monitor = BucketMonitor(
    openai_req_bucket=openai_req_limiter,
    openai_tok_bucket=openai_tok_limiter,
    cohere_req_bucket=cohere_rerank_limiter,
    csv_path="bucket_usage.csv",
)
await monitor.start()   # begins background task

# … run workload …

await monitor.stop()    # graceful shutdown
```
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiofiles


from src.QA_Assistant.token_bucket import AsyncTokenBucket

class BucketMonitor:
    """Background CSV logger for `AsyncTokenBucket` throughput."""

    def __init__(
        self,
        *,
        openai_req_bucket: AsyncTokenBucket,
        openai_tok_bucket: AsyncTokenBucket,
        cohere_req_bucket: AsyncTokenBucket,
        interval: float = 0.5,
        csv_path: str | os.PathLike[str] = "bucket_usage.csv",
        overwrite: bool = True,
    ) -> None:
        self._oa_req = openai_req_bucket
        self._oa_tok = openai_tok_bucket
        self._coh_req = cohere_req_bucket
        self._interval = interval
        self._csv_path = Path(csv_path)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_evt = asyncio.Event()
        self._overwrite = overwrite

    # ─────────────────────────────── public API ────────────────────────────
    async def start(self) -> None:
        """Begin polling in the background."""
        if self._task is not None:
            raise RuntimeError("monitor already running")

        # (Re)create file with CSV header if overwrite or file missing
        if self._overwrite or not self._csv_path.exists():
            async with aiofiles.open(self._csv_path, "w") as f:
                await f.write("time_iso,Cohere_req,OpenAI_req,OpenAI_tok\n")

        self._stop_evt.clear()
        self._task = asyncio.create_task(self._poll_loop(), name="BucketMonitor")

    async def stop(self) -> None:
        """Stop polling. Waits until the background task exits."""
        if self._task is None:
            return  # not running
        self._stop_evt.set()
        await self._task
        self._task = None

    # ────────────────────────────── internals ──────────────────────────────
    async def _poll_loop(self) -> None:
        import aiofiles  # local import to avoid mandatory dependency if unused

        while not self._stop_evt.is_set():
            now_iso = datetime.now().isoformat(timespec="seconds")
            row = [
                now_iso,
                self._remaining(self._coh_req),
                self._remaining(self._oa_req),
                self._remaining(self._oa_tok),
            ]
            async with aiofiles.open(self._csv_path, "a") as f:
                await f.write(",".join(map(str, row)) + "\n")

            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass  # normal periodic wake‑up

    @staticmethod
    def _remaining(bucket: AsyncTokenBucket) -> int:
        """Tokens/requests left in the current sliding window."""
        # Accessing _in_window is safe inside‑process; no public accessor yet.
        return max(0, bucket.capacity - bucket._in_window)
