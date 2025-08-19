""" report_generator.py
~~~~~~~~~~~~~~

"""
from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from openai.types.responses import Response
from src.ReportGenerator.prompts import SYSTEM_PROMPT
from src.gen_ratelimit import gated_call_gen

# ────────────────────────────────────────── helpers ───────────────────────────

def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write("")

# ────────────────────────────────────────── agent ─────────────────────────────

class ReportGenerator:
    """
    """
    MAX_TOOL_ROUNDS: int = 3

    def __init__(
        self,
        *,
        topic,
        client: AsyncAzureOpenAI,
        num: int = 0 
    ) -> None:
        load_dotenv()
        self.topic = topic
        self.client = client

        self.cur_report: str = None
        self.eval_notes: List[str] = []
        self.my_notes: List[str] = []
        self.latest_note: str

        # artefacts
        self.LOG_PATH = f"{os.getenv('REPORT_PATH')}{num}.txt"
        _ensure_file(self.LOG_PATH)
            

    def _log(self, msg: str, *, _file: Optional[str] = None) -> None:
        with open(_file or self.LOG_PATH, "a", encoding="utf-8") as f: f.write(msg)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        start, end = f"<{tag}>", f"</{tag}>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0].strip()
        return None
    
    async def generate_report(self,ir_context,note,eval_) -> list[str]:
        """
        Generates a report based on the topic, IR context, evaluation notes, and previous report
        """
        self.eval_notes.append(note)
        prompt = f"{SYSTEM_PROMPT}\nTopic:\n{self.topic}\nPrevious report: \n{self.cur_report or 'First round no report yet'}\nYour notes:\n{self.serialize_notes(True)}Evaluation notes: \n{self.serialize_notes(False)}\nEvaluation:\n{eval_}\nIR context: \n{ir_context or 'First round no IR context yet'}\n"
        resp: Response = await gated_call_gen(
            prompt = prompt,
            client = self.client,
            temperature=0.25,
        )
        text = resp.output_text
        self._log("\n=========\n")
        self._log(f"Prompt:\n{prompt}\n")
        self._log(f"Response:\n{text}\n")
        self._update_status(content = text)
        return self.cur_report,self.my_notes[-1]


    def serialize_notes(self, mine: bool) -> str:
        notes = self.my_notes if mine else self.eval_notes
        return "\n".join(
            f"{i}. Evaluation note: {note or 'First round no note yet or trouble parsing eval note'}"
            for i, note in enumerate(notes)
        ) + "\n"

    def _update_status(
        self,
        content: str
    ) -> None:
        """
        extracts report and latest note from response
        """
        try: 
            self.cur_report = self._extract_tag(content, "report")
            self.my_notes.append(self._extract_tag(content, "note"))
        except Exception as e:
            print(f"Error extracting report and note: {e}")
            # hope the note is somewhere in there and it will be attached to the report
            self.cur_report = content

