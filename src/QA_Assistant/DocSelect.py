"""
Document‑selection wrapper for the IR project.
"""

import asyncio
import json
from pathlib import Path
from typing import List

JAVA_CLASSPATH = "src/QA_Assistant/Search/lib/*:."

async def select_documents(
    segment_ids: List[str],
    is_segment: bool,
    timeout: int = 120
) -> List[dict]:
    """
    Run the Java DocumentSelection tool and return its JSONL output
    as a list of Python dicts.

    Args
    ----
    segment_ids : up to 4 segment IDs if `is_segment=True`;
                   exactly 1 segment ID if `is_segment=False`.
    is_segment   : True  → return each segment's text
                   False → return full document text built from sliding‑window segments
    timeout      : seconds before the subprocess is killed (default 120 s)

    Returns
    -------
    List[dict] — each dict is what the Java tool printed (parsed from JSON).
    """
    if is_segment and len(segment_ids) > 4:
        raise ValueError("With is_segment=True you may pass up to 4 IDs.")
    if not is_segment and len(segment_ids) == 0:
        raise ValueError("Need at least one ID")

    # ---------- build command ----------
    cmd: list[str] = [
        "java",
        "-cp", JAVA_CLASSPATH,
        "src.QA_Assistant.Search.DocumentSelection",
    ]
    if is_segment:
        cmd.append("--asSegments")
    cmd.extend(segment_ids)

    # ---------- spawn ----------
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None

    if proc.returncode:
        return None

    # ---------- parse JSONL ----------
    jsonl = stdout_bytes.decode("utf-8").strip().splitlines()
    return jsonl


# ---------------- quick test ----------------
async def _demo():
    # single full‑text document
    full = await select_documents(
        ["msmarco_v2.1_doc_24_434728473#0_962031585"],  # any segment ID of the doc
        is_segment=False
    )
    print(full[0])

if __name__ == "__main__":
    asyncio.run(_demo())
