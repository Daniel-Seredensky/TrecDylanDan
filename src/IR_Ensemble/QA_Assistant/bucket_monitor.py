from __future__ import annotations

import asyncio, aiofiles
from datetime import datetime
from typing import Mapping,Optional,Dict
import os
from pathlib import Path
from dotenv import load_dotenv

from src.IR_Ensemble.QA_Assistant.rate_limits import (
    plan_req_limiter,
    plan_tok_limiter,
    global_req_limiter,
    global_tok_limiter,
    assistant_tok_limiters,
    cohere_bucket
)
from src.IR_Ensemble.QA_Assistant.token_bucket import AsyncTokenBucket

class BucketMonitor:
    # ─────────────────────────────── initialisation ──────────────────────────
    def __init__(
        self,
        *,
        plan_req_bucket:AsyncTokenBucket = plan_req_limiter,
        plan_tok_bucket:AsyncTokenBucket = plan_tok_limiter,
        global_req_bucket:AsyncTokenBucket = global_req_limiter,
        global_tok_bucket:AsyncTokenBucket = global_tok_limiter,
        assistant_buckets:Dict[str,AsyncTokenBucket] = assistant_tok_limiters,
        cohere: AsyncTokenBucket = cohere_bucket[0],
        interval: float = 1,
        csv_path: str | os.PathLike[str] = "bucket_usage.csv",
        overwrite: bool = True,
    ) -> None:
        load_dotenv()
        # Core buckets
        self._buckets_static: Dict[str, AsyncTokenBucket] = {
            "Plan_req": plan_req_bucket,
            "Plan_tok": plan_tok_bucket,
            "Global_req": global_req_bucket,
            "Global_tok": global_tok_bucket,
            "Cohere": cohere,
        }

        # The per‑assistant dict (may grow while running)
        self._assistant_buckets: Mapping[str, AsyncTokenBucket] = assistant_buckets

        # IO / scheduling
        self._interval = interval
        base =os.getenv("BUCKET_MONITOR_OUT")
        self._csv_path = Path(base+csv_path)
        self._overwrite = overwrite
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_evt = asyncio.Event()

        # Build the (mutable) column list
        self._columns: list[str] = ["time_iso"] + list(self._buckets_static.keys())
        self._columns += self._assistant_columns()  # any assistants already present

    # ─────────────────────────────── public API ──────────────────────────────
    async def start(self) -> None:
        """Begin polling in the background."""
        if self._task:
            raise RuntimeError("BucketMonitor already running")

        # (Re)create file with CSV header if overwrite or file missing
        if self._overwrite or not self._csv_path.exists():
            async with aiofiles.open(self._csv_path, "w") as f:
                await f.write(",".join(self._columns) + "\n")

        self._stop_evt.clear()
        self._task = asyncio.create_task(self._poll_loop(), name="BucketMonitor")

    async def stop(self) -> None:
        """Stop polling. Waits until the background task exits."""
        if not self._task:
            return
        self._stop_evt.set()
        await self._task
        self._task = None

    # ────────────────────────────── internals ───────────────────────────────
    async def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            # Expand column set if new assistants appeared
            new_cols = self._assistant_columns()
            if any(col not in self._columns for col in new_cols):
                await self._extend_header(new_cols)

            # Compose one CSV row
            now_iso = datetime.now().isoformat(timespec="seconds")
            row_values = {"time_iso": now_iso}

            # Static buckets
            for name, bucket in self._buckets_static.items():
                row_values[name] = self._remaining(bucket)

            # Per‑assistant buckets
            for col in new_cols:
                aid = col.removeprefix("Assistant_").removesuffix("_tok")
                row_values[col] = self._remaining(self._assistant_buckets[aid])

            # Ensure column order matches header
            row = [str(row_values.get(col, "")) for col in self._columns]
            async with aiofiles.open(self._csv_path, "a") as f:
                await f.write(",".join(row) + "\n")

            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass  # periodic wake‑up

    # ───────────────────────── header / helpers ─────────────────────────────
    def _assistant_columns(self) -> list[str]:
        """Return list of 'Assistant_<id>_tok' cols, sorted for stability."""
        return [f"Assistant_{aid}_tok" for aid in sorted(self._assistant_buckets.keys())]

    async def _extend_header(self, new_cols: list[str]) -> None:
        """Add new assistant columns to the CSV header (rewrites the file)."""
        added = [col for col in new_cols if col not in self._columns]
        if not added:
            return
        self._columns += added

        # Read existing contents (if any) ➜ prepend new header ➜ rewrite
        if self._csv_path.exists():
            async with aiofiles.open(self._csv_path, "r") as f:
                existing = await f.read()
        else:
            existing = ""

        async with aiofiles.open(self._csv_path, "w") as f:
            await f.write(",".join(self._columns) + "\n")
            if existing:
                # Skip the old header line (everything after first '\n')
                await f.write(existing.split("\n", 1)[-1])

    @staticmethod
    def _remaining(bucket: AsyncTokenBucket) -> int:
        """Tokens/requests left in the current sliding window."""
        return max(0, bucket.capacity - bucket.current_load())
