from __future__ import annotations

import os
from openai import AsyncOpenAI
from openai.types.responses import Response
import json
from typing import List, Dict, Any, Optional
from src.ReportEvaluator.prompts import SYSTEM_PROMPT
from enum import Enum


class EvalStatus(Enum):
        """
        Enum for the evaluation status of a report.
        """
        PASS = "PASS"
        FAIL = "FAIL"
        INCOMPLETE = "INCOMPLETE"

# TODO: create a method that aggregates scores and then if above a threshold set self.status to PASS
# TODO: 
class ReportEvaluator:

    """
    Uses LLM evaluate a report, provide feedback, and generate IR questions.
    """
    def __init__(self, client: AsyncOpenAI, topic: str):
        self.client = client
        self.topic = topic
        self.my_notes = []
        self.gen_notes = []
        self._ensure_file(os.getenv("EVAL_PATH"))
        self.status = EvalStatus.INCOMPLETE
        self.questions: str
    
    async def evaluate(
        self,
        report: List[Dict[str, str]],
        ir_context: Dict[str, Any],
        generator_comment: str = ""
    ) -> Dict[str, Any]:
        """
        Calls LLM to evaluate the report and return the evaluation contract.
        """
        self.gen_notes.append(generator_comment)

        prompt = SYSTEM_PROMPT + f"\nTopic document:\n{self.topic}\n \nReport:\n{report}\nIR Context:\n{ir_context}\nGenerator Comments:\n{self.serialize_notes(False)}\n Your Comments:\n{self.serialize_notes(True)}"
        response: Response = await self.client.responses.create(
            model="gpt-4.1",
            input = prompt,
            temperature=0.2,
        )
        self._update_status(response.output_text)
        return self.my_notes[-1],self.questions

        
    
    def serialize_notes(self, mine: bool) -> str:
        notes = self.my_notes if mine else self.gen_notes
        return "\n".join(
            f"{i}. Evaluation note: {note or 'First round no note yet or trouble parsing eval note'}"
            for i, note in enumerate(notes)
        ) + "\n"
    
    def _log(self, msg: str, *, _file: Optional[str] = None) -> None:
        with open(_file or self.LOG_PATH, "a", encoding="utf-8") as f: f.write(msg)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        start, end = f"<{tag}>", f"</{tag}>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0].strip()
        return None
    
    def _ensure_file(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: f.write("")

    def _update_status(self, content):
        """
        Answer contract
            <cot> Plan out your evaluation here, note the good, the bad, and relevant planning steps.</cot>
            <note> Your note to the report generator goes here. </note>
            <ir> 
            {
            questions: [
            {
                "question": <question1>,
                "context": <context from the document that might be needed to answer the question>,
            },
            {
                "question": <question2>,
                "context": <context from the document that might be needed to answer the question>,
            },
            ...
            ]
            }
            </ir>
            <eval> 
            {
            "coverage": 4,
            "accuracy": 5,
            "citation_quality": 3,
            "style": 4,
            "prioritization": 4,
            "completeness": 3,
            }
            </eval>
        """
        note,self.questions = self._extract_tag(content, "note"), self._extract_tag(content, "ir")
        self.my_notes.append(note)
        eval = self._extract_tag(content, "eval")
        self._update_eval(eval)

