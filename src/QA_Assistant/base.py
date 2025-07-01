"""base.py
Base class for an async agent that coordinates plan–search–answer loops using the
Azure OpenAI **Responses API**. Manages internal state specifics of question eval agent

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
   b. `await update_answer(selected_segments)` → updates structured answer.
3. Caller ends when every question is marked `finished:true` or after 5 rounds.

`run()` orchestration is intentionally left for subclasses.
"""
from __future__ import annotations

import os
import json
import uuid
from enum import Enum
from typing import Any, Awaitable, Dict, List, Optional
import traceback

import aiofiles, asyncio
from asyncinit import asyncinit
from openai import AsyncAzureOpenAI
from openai.types.responses import Response
from src.QA_Assistant.answer_contracts import (
    SEARCH_CONTRACT,
    SELECT_CONTRACT,
    PLAN_CONTRACT,
    UPDATE_CONTRACT,
    FINAL_CONTRACT
)
from src.QA_Assistant.Searcher import search
from src.QA_Assistant.DocSelect import select_documents
from src.QA_Assistant.rate_limits import gated_response, LoopStage

# ───────────────────────────────────────── constants ──────────────────────────

MODEL_NAME: str = os.getenv("OPENAI_RESPONSES_MODEL")

BM25_RESULTS_PATH: str = os.getenv("BM25_RESULTS_PATH")


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
    """
    Plan / search / answer‑update skeleton.
    """
    MAX_TOOL_ROUNDS: int = 5

    async def __init__(
        self,
        *,
        questions: List[str],
        client: AsyncAzureOpenAI,
    ) -> None:
        self.questions = questions
        self.client = client

        # Responses API chains via response‑id

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
        tasks = [_ensure_file(p) for p in (self.convo_path, self.tools_path)]
        await asyncio.gather(*tasks)
            
        await self._log(f"Agent {self.agent_id} created)\n")

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
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in self.history)

    # ────────────────────────── public API: PLAN stage ───────────────────────

    async def get_plan_for_questions(self) -> str:
        """
        Gets the plan for the given questions.
        """
        prompt = PLAN_CONTRACT + f"\n\n{self.questions}"
        self._record("user", prompt)

        resp: Response = await gated_response(assistant_id=self.agent_id,
                                    client=self.client, 
                                    prompt=prompt,
                                    stage = LoopStage.PLAN_CALL)
        content = resp.output_text
        self._record("assistant",content)
        self.plan = self._extract_tag(content, "answer")

        await self._log(self._serialise_history())
        await self._log("\n----------------------\n")
        await self.reset_logical_thread()
        

    # ───────────────────── public API: INFO (search + select) ─────────────────

    async def get_info(self, *, first_round: bool) -> str:
        """Runs **two** LLM turns:
        1. Ask for Search tool calls → dispatch them.
        2. Feed back metadata → ask for Select‐Documents tool call → dispatch.
        Returns the *selected‑segments JSON* produced by the select_documents tool.
        """

        if not self.plan:
            raise RuntimeError("Must call get_plan_for_questions() first")

        # Build shared context fragment
        if first_round:
            context_block = "\n\n".join([
                "<plan>" + self.plan + "</plan>",
                "<questions>" + self.questions + "</questions>",
            ])
        else:
            context_block = "\n\n".join([
                "<plan>" + self.plan + "</plan>",
                "<answer>" + (self.full_answer or "") + "</answer>",
            ])

        # ── Ask for SEARCH tool calls ────────────────────────────────────
        content = SEARCH_CONTRACT + context_block
        self._record("user", content)
        anchor: Response = await gated_response(assistant_id=self.agent_id,
                                      client=self.client,
                                      prompt=content,
                                      stage = LoopStage.TOOL_CALL)
        LoopStage.TOOL_CALL.value[0]["previous_response_id"] = anchor.id # update for next tool call 
        search_calls = anchor.output_text
        self._record("assistant",search_calls)
        await self._log(f"\n------- SEARCH CALLS------\n{self._serialise_history()}")

        # Parse & dispatch each search call
        try:
            # Search calls answer contract 
            # """
            # <answer>
            #     {
            #         "searches":[
            #             {
            #                 "queries": [
            #                     <query1>,
            #                     <query2>,
            #                     ...
            #                 ],
            #                 "master_query": "master_query"
            #             },
            #             ...
            #         ] 
            #     }
            # <answer>
            # """
            search_calls = self._extract_tag(search_calls, "answer")
            search_calls = json.loads(search_calls)
            search_calls = search_calls["searches"]
            tasks = [self._dispatch_tool(search, 
                                        **{"queries": call["queries"],
                                        "master_query": call["master_query"]}) 
                                         for call in search_calls]
                     
            search_results = "\n".join([json.dumps(result) 
                                        for result in await asyncio.gather(*tasks)])
        except Exception as e:  # noqa: BLE001 – log & rethrow
            traceback.print_exc()
            raise ValueError(f"LLM produced invalid search‑call JSON: {e}\n{search_calls}")

        # ── Ask for SELECT_DOCUMENTS tool call ───────────────────────────
        content = SELECT_CONTRACT + "\n\n<search_metadata>" + search_results + "</search_metadata>"
        await self._log(f"\n------TOOL RESULTS-------\n{content}")
        resp_select: Response = await gated_response(assistant_id=self.agent_id,
                                            client=self.client,
                                            prompt=content,
                                            stage = LoopStage.TOOL_CALL,
                                            context = self._serialise_history())
        select_calls = resp_select.output_text
        await self._log(f"\n-SELECT CALLS (NOT PERSISTED IN LOGICAL THREAD)-\n{select_calls}")

        # Dispatch select_documents will vanish from run context because selection will
        #  not be inherently useful later, search is useful for context persistence
        try:
            # """
            # <answer>
            # {
            #   "selections":[
            #       <segment_id1>,
            #       <segment_id2>,
            #       ...
            #   ]
            # }
            # </answer>
            # """
            select_calls = self._extract_tag(select_calls, "answer")
            select_calls = json.loads(select_calls)
            select_calls = select_calls["selections"]
            tasks = [self._dispatch_tool(select_documents,
                                          **{"segment_ids": [segment_id], "is_segment": True})
                                          for segment_id in select_calls]
            selected_segments = "\n".join(json.dumps(result) 
                                          for result in await asyncio.gather(*tasks))
        except Exception as e:  # noqa: BLE001 - log & rethrow
            traceback.print_exc()
            raise ValueError(f"LLM produced invalid select_documents JSON: {e}\n{select_calls}")

        await self._log(f"\n----RESULTS----\n{selected_segments}")

        # chain response id for next update_answer
        LoopStage.UPDATE_CALL.value[0]["previous_response_id"] = anchor.id

        return selected_segments

    # ─────────────────────── public API: ANSWER update ───────────────────────

    async def update_answer(self, tool_outputs: str) -> str:
        content = UPDATE_CONTRACT + "\n\n<selected_segments>" + tool_outputs + "</selected_segments>"
        self._record("user", content)

        resp: Response = await gated_response(assistant_id=self.agent_id,
                                    client=self.client,
                                    prompt=content,
                                    stage = LoopStage.UPDATE_CALL,
                                    context = self._serialise_history())
        raw = resp.output_text
        self._record("assistant", raw)
        self.full_answer = self._extract_tag(raw, "answer") or raw
        # Set the previous response for final call
        # If it is not the final call the logical thread will be reset, making all stages previous id none
        LoopStage.FINAL_CALL.value[0]["previous_response_id"] = resp.id

        await self._log(f"\n==== UPDATE PROMPT ====\n{content}\n==== ANSWER UPDATE ====\n{raw}\n")
        self._update_status()
        return raw

    # ─────────────────────────────── utilities ───────────────────────────────

    async def reset_logical_thread(self) -> None:
        print("Resetting logical thread")
        self.history.clear()
        await self._log("\n―――― Logical thread reset ――――\n")
        for stage in LoopStage:
            stage.value[0]["previous_response_id"] = None


    async def _dispatch_tool(self, tool: Awaitable, **kwargs) -> str:
        if tool.__name__ == "search":
            results = await search(**kwargs,agentId=self.agent_id)
            payload = {"call":"search","kwargs": kwargs,"results":results}
            await self._log(f"\n----TOOL CALL----\n{payload}", _file=self.tools_path)
            return {"search": json.dumps(kwargs)[:150], "results":results}
        results = await select_documents(**kwargs)
        payload = {"call":"select_documents","kwargs": kwargs,"results":results}
        await self._log(f"\n----TOOL CALL----\n{payload}", _file=self.tools_path)
        return {"segment_id:": kwargs["segment_ids"], "segment_text": results}

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        start, end = f"<{tag}>", f"</{tag}>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0].strip()
        return None

    def _update_status(self, is_summary = False, content = None) -> None:
        # Check if we are updating summary first
        if is_summary :
            self.summary = self._extract_tag(text=content, tag="summary") if content is not None else ""
            return 
        # Not summary, updating answer status
        if not self.full_answer:
            self.status = QAStatus.NO_ANSWER
            return
        try:
            payload = json.loads(self.full_answer)
            flags = [q["finished"] for q in payload["questions"] or []]
            self.status = (
                QAStatus.FINISHED if all(flags)
                else QAStatus.PARTIAL if any(flags)
                else QAStatus.NO_ANSWER
            )
        except Exception:
            self.status = QAStatus.PARTIAL
            traceback.print_exc()
            raise ValueError("Invalid answer format")

    # ───────────────────────────── placeholder hooks ─────────────────────────

    async def force_final_prompt(self) -> str:
        self._record("user", FINAL_CONTRACT)
        await self._log(f"\n-----FORCED FINAL-----\n{FINAL_CONTRACT}")
        resp: Response = await gated_response(assistant_id=self.agent_id,
                                    client=self.client,
                                    prompt=FINAL_CONTRACT,
                                    stage = LoopStage.FINAL_CALL,
                                    context = self._serialise_history()
                                    )
        answer = resp.output_text
        await self._log("\n==== FINAL SUMMARY ====\n" + answer + "\n")
        self._update_status(is_summary=True, content=answer)

    async def run(self) -> None:  # pragma: no cover
        raise NotImplementedError

# TODO: write a one loop test PLAN -> GET INFO -> UPDATE -> FORCE FINAL PROMPT
async def test(client,questions):
    agent: BaseAgent = await BaseAgent(client=client, questions=questions)
    await agent.get_plan_for_questions()
    segments = await agent.get_info(first_round=True)
    await agent.update_answer(segments)
    await agent.force_final_prompt()