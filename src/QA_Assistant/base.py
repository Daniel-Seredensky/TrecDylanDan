"""base.py
Base class for an async agent that coordinates plan–search–answer loops using the
Azure OpenAI **Responses API**.

High‑level lifecycle (orchestrated externally)
──────────────────────────────────────────────
1. `await get_plan_for_questions()` → stores **self.plan** and resets logical
   thread.
2. Up to **MAX_TOOL_ROUNDS**
   a. `await get_info(first_round)` →
      • asks model for **search** tool calls
      • dispatches them and feeds back meta
      • asks model for **select_documents** tool call
      • dispatches it and returns the selected‑segment JSON
   b. `await update_answer(selected_segments_json)` → updates structured answer.
3. Caller ends when every question is marked `finished:true` or after 5 rounds.

`run()` orchestration is intentionally left for subclasses.
"""
from __future__ import annotations

import os
import json
import uuid
from enum import Enum
from typing import Any, Awaitable, Dict, List, Optional

import aiofiles, asyncio
from asyncinit import asyncinit
from openai import AsyncAzureOpenAI
from answer_contracts import (
    SEARCH_CONTRACT,
    SELECT_CONTRACT,
    PLAN_CONTRACT,
    UPDATE_CONTRACT,
    FINAL_CONTRACT
)

# ───────────────────────────────────────── constants ──────────────────────────

MAX_TOOL_ROUNDS: int = 5
MODEL_NAME: str = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")

BM25_RESULTS_PATH: str = os.getenv("BM25_RESULTS_PATH", "DerivedData/QA")


class QAStatus(str, Enum):
    NO_ANSWER = "no_answer"
    PARTIAL = "partial_answer"
    FINISHED = "finished"


# ────────────────────────────────────────── helpers ───────────────────────────

async def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, "a", encoding="utf-8"):
        pass

# ────────────────────────────────────────── agent ─────────────────────────────

@asyncinit
class BaseAgent:
    """Plan / search / answer‑update skeleton.

    Subclasses may override prompt fragments, parsing, or `_dispatch_tool()`
    to plug in real search + selection back‑ends.
    """

    async def __init__(
        self,
        *,
        questions: List[str],
        client: AsyncAzureOpenAI,
    ) -> None:
        self.questions = questions
        self.client = client

        # Responses API chains via response‑id
        self.prev_resp_id: Optional[str] = None

        self.history: List[Dict[str, str]] = []  # mirrors message list for token‑estimation
        self.agent_id = str(uuid.uuid4())
        self.status: QAStatus = QAStatus.NO_ANSWER

        self.plan: Optional[str] = None
        self.full_answer: Optional[str] = None
        self.summary: Optional[str] = None

        # artefacts
        self.results_path = os.path.join(BM25_RESULTS_PATH, self.agent_id)
        self.convo_path = os.path.join(self.results_path, "Convo.txt")
        self.tools_path = os.path.join(self.results_path, "Tools.txt")
        self.thread_path = os.path.join(self.results_path, "Thread.txt")
        tasks = [_ensure_file(p) for p in (self.convo_path, self.tools_path, self.thread_path)]
        await asyncio.gather(tasks)
            
        await self._log(f"Agent {self.agent_id} created ({len(questions)} Qs)\n")

    # ───────────────────────────── logging & history ─────────────────────────

    async def _log(self, msg: str, *, _file: Optional[str] = None) -> None:
        async with aiofiles.open(_file or self.convo_path, "a", encoding="utf-8") as f:
            await f.write(msg)

    def _record(self, role: str, content: str) -> None:
        """Mirror a message in local history for token counting."""
        self.history.append({"role": role, "content": content})

    def _serialise_history(self) -> str:
        """Return *exact* plain‑text mirror of what the backend tokenises.

        The Responses API internally tokenises a conversation as the raw list we
        pass in `input=[{"role":…, "content":…}, …]`.  For parity with previous
        Assistant‑style accounting we keep the same `<|role|>\n…` representation
        so downstream callers can reuse their Tiktoken estimators unchanged.
        """
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in self.history) + "<|assistant|>"

    # ────────────────────────── public API: PLAN stage ───────────────────────

    async def get_plan_for_questions(self) -> str:
        """
        Gets the plan for going about solving the questions
        """
        prompt = PLAN_CONTRACT + f"\n\n{self.questions}"
        self._record("user", prompt)

        resp = await self._create_llm_response(messages=prompt)
        content = resp.choices[0].message.content
        self._record("assistant",content)
        self.plan = self._extract_tag(content, "plan")

        await self._log(self._serialise_history)
        await self._log("\n----------------------\n")
        await self.reset_logical_thread()
        return self.plan or ""

    # ───────────────────── public API: INFO (search + select) ─────────────────

    async def get_info(self, *, first_round: bool) -> str:
        """Runs **two** LLM turns:
        1. Ask for Search tool calls → dispatch them.
        2. Feed back metadata → ask for Select‐Documents tool call → dispatch.
        Returns the *selected‑segments JSON* produced by the select_documents tool.
        Sets `self.prev_resp_id` to the _last_ assistant message so `update_answer`
        can reference it.
        """

        if not self.plan:
            raise RuntimeError("Must call get_plan_for_questions() first")

        # Build shared context fragment
        if first_round:
            context_block = "\n\n".join([
                "<plan>" + self.plan + "</plan>",
                "<questions>" + "\n".join(self.questions) + "</questions>",
            ])
        else:
            context_block = "\n\n".join([
                "<plan>" + self.plan + "</plan>",
                "<answer>" + (self.full_answer or "") + "</answer>",
            ])

        # ── Ask for SEARCH tool calls ────────────────────────────────────
        content = SEARCH_CONTRACT + context_block
        self._record("user", content)
        anchor = await self._create_llm_response(messages=content)
        search_calls = anchor.choices[0].message.content.strip()
        self._record("assistant",search_calls)
        self._log(f"\n------- SEARCH CALLS------\n{self._serialise_history()}")
                  

        # Parse & dispatch each search call
        try:
            search_calls = json.loads(search_calls)
        except Exception as e:  # noqa: BLE001 – log & rethrow
            raise ValueError(f"LLM produced invalid search‑call JSON: {e}\n{search_calls}")
        
        # TODO: update result parsing name it search_results
        search_meta_list = []
        for call in search_calls:
            if call.get("name") != "search":
                continue  # ignore unexpected
            args = call.get("args", {})
            meta_json = await self._dispatch_tool("search", args)
            search_meta_list.append(meta_json)
            self._record("tool", meta_json)

        # ── Ask for SELECT_DOCUMENTS tool call ───────────────────────────
        content = SELECT_CONTRACT + "\n\n<search_metadata>" + search_results + "</search_metadata>"
        self._record("user", content)
        resp_select = await self._create_llm_response(
            messages=content,
            previous_response_id=anchor.id,
            tools=self.tools,
        )
        select_calls = resp_select.choices[0].message.content.strip()
        self._log(f"\n-SELECT CALLS (NOT PERSISTED IN LOGICAL THREAD)-\n{select_calls}")

        # Dispatch select_documents will vanish from run context because selection will
        #  not be inherently useful later, search is useful for context persistence
        try:
            select_calls = json.loads(select_calls)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"LLM produced invalid select_documents JSON: {e}\n{select_calls}")

        # TODO: update result parsing
        selected_segments_json = await self._dispatch_tool("select_documents", select_calls.get("args", {}))
        self.log(f"\n----RESULTS----\n{selected_segments_json}")

        # chain prev_resp_id for update_answer
        self.prev_resp_id = anchor.id

        return selected_segments_json

    # ─────────────────────── public API: ANSWER update ───────────────────────

    async def update_answer(self, tool_outputs: str) -> str:
        if not self.prev_resp_id:
            raise RuntimeError("get_info() must precede update_answer()")

        content = UPDATE_CONTRACT + "\n\n<selected_segments>" + tool_outputs + "</selected_segments>"
        self._record("user", content)

        resp = await self._create_llm_response(
            messages=content,
            previous_response_id=self.prev_resp_id,
        )
        raw = resp.choices[0].message.content
        self.full_answer = self._extract_tag(raw, "answer") or raw
        await self._log(f"\n==== UPDATE PROMPT ====\n{content}\n==== ANSWER UPDATE ====\n{raw}\n")
        self._update_status()
        return raw

    # ─────────────────────────────── utilities ───────────────────────────────

    async def _create_llm_response(self, *, messages: str, **kwargs: Any):
        return await self.client.responses.create(
            model=MODEL_NAME,
            input=messages,
            stream=False,
            **kwargs,
        )

    async def reset_logical_thread(self) -> None:
        self.history.clear()
        self.prev_resp_id = None
        await self._log("\n―――― Logical thread reset ――――\n")

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        start, end = f"<{tag}>", f"</{tag}>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0].strip()
        return None

    def _update_status(self, is_summary = False, content = None) -> None:
        if not self.full_answer:
            self.status = QAStatus.NO_ANSWER
            return
        # Check if we are updating summary first
        if is_summary :
            self.summary = self._extract_tag(text=content, tag="summary") if content is not None else ""
            return 
        try:
            payload = json.loads(self.full_answer)
            flags = [q.get("finished") for q in payload.get("questions", [])]
            self.status = (
                QAStatus.FINISHED if all(flags)
                else QAStatus.PARTIAL if any(flags)
                else QAStatus.NO_ANSWER
            )
        except Exception:
            self.status = QAStatus.PARTIAL

    # ─────────────────────────── tool dispatch helper ────────────────────────
    # TODO: update _dispatch tool to use the new results
    async def _dispatch_tool(self, name: str, payload: Dict[str, Any]) -> str:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered")
        fn = self.tools[name]
        result = await fn(payload)  # always async
        output = result[:2000]
        await self._log(json.dumps({"tool": name, "payload": payload, "output": output}, indent=2) + "\n", _file=self.tools_path)
        return output

    # ───────────────────────────── placeholder hooks ─────────────────────────

    async def _force_final_prompt(self) -> str:
        self._record("user", FINAL_CONTRACT)
        self._log(f"\n-----FORCED FINAL-----\n{FINAL_CONTRACT}")
        resp = await self._create_llm_response(messages=FINAL_CONTRACT, previous_response_id = self.prev_resp_id)
        answer = resp.choices[0].message.content
        await self._log("\n==== FINAL SUMMARY ====\n" + answer + "\n")
        self._update_status(is_summary=True, content = answer)

    async def run(self) -> None:  # pragma: no cover
        raise NotImplementedError

# TODO: write a one loop test PLAN -> GET INFO -> UPDATE -> FORCE FINAL PROMPT
async def test(client,questions):
