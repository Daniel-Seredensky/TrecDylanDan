"""
token_bucket.py
~~~~~~~~~~~~~~~~
A minimal, asyncio‑friendly sliding‑window rate‑ / token‑limiter.

• Works with *weights* (so you can reserve N tokens, not just 1 “request”).
• Guarantees the window is respected even under heavy concurrency.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Tuple


class AsyncTokenBucket:
    """
    Sliding‑window bucket (capacity units per *window* seconds).

    >>> tok_bucket = AsyncTokenBucket(50_000, window=60)
    >>> async with tok_bucket.acquire(3_000):
    ...     await call_openai()
    """

    def __init__(self, capacity: int, window: float = 60.0) -> None:
        self.capacity: int = capacity
        self.window: float = window

        self._events: Deque[Tuple[float, int]] = deque()   # (timestamp, weight)
        self._in_window: int = 0                           # running total
        self._lock = asyncio.Lock()

    # ────────────────────────────────────────── helpers ──────────────────────
    def _purge_old(self, now: float) -> None:
        """Drop events that have aged out of the sliding window."""
        w = self.window
        while self._events and (now - self._events[0][0] >= w):
            _, weight = self._events.popleft()
            self._in_window -= weight

    async def _wait_for_slot(self, weight: int) -> None:
        """
        Block until *weight* units fit inside the bucket.
        Uses a sliding window, so we may have to sleep.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_old(now)

                if self._in_window + weight <= self.capacity:
                    # reserve the slot(s)
                    self._events.append((now, weight))
                    self._in_window += weight
                    return

                # not enough room → compute earliest expiry
                oldest_ts, _ = self._events[0]
                sleep_for = (self.window - (now - oldest_ts)) + 1e-3  # tiny safety pad

            await asyncio.sleep(sleep_for)

    # ────────────────────────────── public API ───────────────────────────────
    @asynccontextmanager
    async def acquire(self, weight: int = 1):
        """
        Async context manager that **blocks** until *weight* units are available
        and then reserves them for the duration of the context.

        Nothing needs to be “released”; units simply age‑out after *window* sec.
        """
        await self._wait_for_slot(weight)
        try:
            yield
        finally:
            # Nothing to do – the reservation naturally expires after <window>.
            pass

    async def credit(self, weight: int) -> None:
        """Return *weight* units immediately to the current window."""
        if weight <= 0:
            return
        async with self._lock:          # (lock is an `asyncio.Lock`; use `async with` if you prefer)
            self._in_window = max(0, self._in_window - weight)
