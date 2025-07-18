from __future__ import annotations

"""
AsyncTokenBucket
~~~~~~~~~~~~~~~~
A sliding‑window rate / token limiter that now **tags every reservation with a
unique event‑id** so callers can refund exactly the tokens they reserved – even
when sharing a global bucket across many workers.

Usage
-----
````python
bucket = AsyncTokenBucket(50_000, window=60)

async with bucket.acquire(3_000) as event_id:
    response = await call_openai()
    # ... decide you need to refund 1_200 …
    await bucket.credit_by_id(event_id, 1_200)
````

If you don’t capture the `event_id`, you can still use the legacy `credit` API
(which walks the deque newest‑to‑oldest).
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Optional, Tuple


class AsyncTokenBucket:
    """Sliding‑window bucket (capacity units per *window* seconds)."""

    # ─────────────────────────── construction ────────────────────────────
    def __init__(self, capacity: int, window: float = 60.0) -> None:
        self.capacity: int = capacity
        self.window: float = window

        # Each event: (timestamp, weight, event_id)
        self._events: Deque[Tuple[float, int, str]] = deque()
        self._in_window: int = 0           # running total in the current window
        self._lock = asyncio.Lock()

        # Simple monotonically‑increasing counter for ids (stringified ints)
        self._next_id: int = 0

    # ───────────────────────── internal helpers ──────────────────────────
    def _purge_old(self, now: float) -> None:
        """Drop events that have aged out of the sliding window."""
        w = self.window
        while self._events and (now - self._events[0][0] >= w):
            _, weight, _ = self._events.popleft()
            self._in_window -= weight

    async def _reserve(self, weight: int) -> str:
        """Block until *weight* units fit, then record and return an `event_id`."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_old(now)

                if self._in_window + weight <= self.capacity:
                    event_id = str(self._next_id)
                    self._next_id += 1

                    self._events.append((now, weight, event_id))
                    self._in_window += weight
                    return event_id

                # Earliest expiry → how long we need to wait
                oldest_ts, _, _ = self._events[0]
                sleep_for = (self.window - (now - oldest_ts)) + 1 # +1 to be safe
            await asyncio.sleep(sleep_for)

    # ───────────────────────────── public API ────────────────────────────
    @asynccontextmanager
    async def acquire(self, weight: int = 1):
        """Async CM that reserves *weight* units and yields the **event_id**."""
        event_id = await self._reserve(weight)
        try:
            yield event_id
        finally:
            # Nothing to do – reservation expires naturally after <window> sec.
            pass

    async def credit_by_id(self, event_id: str, weight: Optional[int] = None) -> None:
        """Refund up to *weight* tokens from the reservation identified by *event_id*.

        * If *weight* is ``None`` → refund the entire event.
        * If the event has already fully expired → no‑op.
        """
        async with self._lock:
            now = time.monotonic()
            self._purge_old(now)

            # Scan newest‑to‑oldest for efficiency
            for idx in range(len(self._events) - 1, -1, -1):
                ts, w, eid = self._events[idx]
                if eid != event_id:
                    continue

                refund = w if weight is None else min(weight, w)
                self._in_window -= refund

                if refund == w:
                    # Drop the whole event
                    del self._events[idx]
                else:
                    # Shrink the event
                    self._events[idx] = (ts, w - refund, eid)
                break

            # Safety net
            self._in_window = max(0, self._in_window)

    # ───────────────────────────── introspection ──────────────────────────
    def current_load(self) -> int:
        """Return the total units currently in the sliding window."""
        return self._in_window
