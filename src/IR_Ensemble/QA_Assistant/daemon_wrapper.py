import asyncio, json, os, re, signal, secrets, time
from pathlib import Path
from typing import List, Any, Dict, Optional


JAVA_CLASSPATH = "src/IR_Ensemble/QA_Assistant/Search/lib/*:."

# ───────────────────── framing helpers ──────────────────────
_HEADER_DELIM = b"\r\n\r\n"
_CL_RE        = re.compile(rb"Content-Length:\s*(\d+)", re.I)

async def _read_frame(reader: asyncio.StreamReader) -> bytes:
    """
    Read *one* framed message: headers ending with <CRLF><CRLF>,
    then exactly N bytes, where N is the Content‑Length value.
    """
    hdr = await reader.readuntil(_HEADER_DELIM)
    m   = _CL_RE.search(hdr)
    if not m:
        raise RuntimeError("Missing Content‑Length header")
    length = int(m.group(1))
    return await reader.readexactly(length)

def _encode_frame(payload: bytes) -> bytes:
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode()
    return header + payload

# ───────────────────── daemon wrapper ───────────────────────
class JVMDaemon:
    """
    Singleton manager for the long‑running Daemon JVM.

    Public async API
    ----------------
    • JVMDaemon.run_bm25_search(queries, out_path)
    • JVMDaemon.select_documents(segment_ids, is_segment)
    • JVMDaemon.stop()
    """

    _instance = None

    # ───────── singleton plumbing ─────────
    def __new__(cls, *a, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, java_classpath: str = JAVA_CLASSPATH, *,
                 jvm_opts: List[str] | None = None,
                 loop: asyncio.AbstractEventLoop | None = None) -> None:
        if hasattr(self, "_init"):
            return
        self._init = True

        self._classpath = java_classpath
        self._jvm_opts  = jvm_opts or []
        self._loop      = loop or asyncio.get_event_loop()

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock   = asyncio.Lock()           # serialise writes only
        self._start_lock   = asyncio.Lock()           
        self._pending: Dict[str, asyncio.Future] = {}

        self._reader: Optional[asyncio.Task]  = None
        self._stderr_task: Optional[asyncio.Task]  = None 

    # ───────── lifecycle helpers ──────────
    async def _start(self) -> None:
        async with self._start_lock:                  # <── ensures single creator
            if self._proc and self._proc.returncode is None:
                return                                # already running

            self._proc = await asyncio.create_subprocess_exec(
                "java",
                "-cp", self._classpath,
                *self._jvm_opts,
                "src.IR_Ensemble.QA_Assistant.Search.SearcherDaemon",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )

            # stderr drainer – create exactly once
            if not self._stderr_task or self._stderr_task.done():
                self._stderr_task = self._loop.create_task(
                    self._drain_stderr(), name="SearcherDaemon‑stderr"
                )

    async def _submit(self, call: str, params: List[str | Path]) -> Any:
        await self._start()
        if not self._reader:
            self._start_reader()     # no await – runs forever

        req_id  = secrets.token_hex(4)
        fut     = self._loop.create_future()
        self._pending[req_id] = fut

        body = json.dumps({"id": req_id, "call": call, "params": [str(p) for p in params]}
                          ).encode()
        frame = _encode_frame(body)

        async with self._lock:
            self._proc.stdin.write(frame)
            await self._proc.stdin.drain()

        return await fut   # resolves when the reader sees the matching id

    # ───────── background reader ──────────
    def _start_reader(self) -> None:
        async def _loop():
            try:
                while True:
                    raw = await _read_frame(self._proc.stdout)
                    resp = json.loads(raw.decode())

                    fut = self._pending.pop(resp.get("id"), None)
                    if not fut or fut.done():
                        continue

                    if resp.get("status") == 0:
                        # success: prefer resultJson ‑> result ‑> full resp
                        fut.set_result(
                            resp.get("resultJson") or
                            resp.get("result")     or
                            resp
                        )
                    else:
                        fut.set_result(resp)
            except asyncio.IncompleteReadError:
                # JVM exited; fail all pending futures
                for f in self._pending.values():
                    if not f.done():
                        f.set_exception(RuntimeError("JVM daemon closed stdout"))
            except Exception as e:
                for f in self._pending.values():
                    if not f.done():
                        f.set_exception(e)

        self._reader = asyncio.create_task(_loop(), name="daemon‑reader")

    async def _drain_stderr(self):
        if not self._proc or not self._proc.stderr:
            return
        async for line in self._proc.stderr:
            print("[JVM]", line.decode().rstrip())

    # ───────── public class‑level API ─────────
    @classmethod
    async def run_bm25_search(cls, queries: List[str], out_path: Path) -> None:
        daemon = cls()
        await daemon._submit("search", [*queries, out_path])

    @classmethod
    async def select_documents(
        cls,
        segment_ids: List[str],
        is_segment: bool,
    ) -> List[dict]:
        flag   = ["--asSegments"] if is_segment else []
        daemon = cls()
        raw    = await daemon._submit("selectDocuments", [*flag, *segment_ids])

        # `raw` is already the JSON string returned by DocumentSelection.run()
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else [data]
        except TypeError:
            return str(raw)  # return raw if JSON decode fails
        except json.JSONDecodeError as e:
            return str(raw)

    # ───────── optional shutdown ──────────
    @classmethod
    async def stop(cls, *, graceful: float = 5.0, term: float = 3.0) -> None:
        daemon = cls._instance
        if not daemon or not daemon._proc:
            return

        if daemon._proc.stdin and not daemon._proc.stdin.is_closing():
            daemon._proc.stdin.close()

        try:
            await asyncio.wait_for(daemon._proc.wait(), timeout=graceful)
            return
        except asyncio.TimeoutError:
            daemon._proc.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(daemon._proc.wait(), timeout=term)
        except asyncio.TimeoutError:
            daemon._proc.kill()
            await daemon._proc.wait()
