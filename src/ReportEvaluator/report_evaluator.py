from __future__ import annotations

import os, json 
from openai import AsyncOpenAI
from openai.types.responses import Response
from typing import List, Dict, Any, Optional
from src.ReportEvaluator.prompts import SYSTEM_PROMPT
from enum import Enum
from dotenv import load_dotenv
from src.gen_ratelimit import gated_call_gen

class EvalStatus(Enum):
        """
        Enum for the evaluation status of a report.
        """
        PASS = "PASS"
        FAIL = "FAIL"
        INCOMPLETE = "INCOMPLETE"

class ReportEvaluator:

    """
    Uses LLM evaluate a report, provide feedback, and generate IR questions.
    """
    def __init__(self, client: AsyncOpenAI, topic: str, num: int = 0):
        load_dotenv()
        self.client = client
        self.topic = topic
        self.my_notes = []
        self.gen_notes = []
        self.LOG_PATH = f"{os.getenv('EVAL_PATH')}{num}.txt"
        self._ensure_file(self.LOG_PATH)
        self.status = EvalStatus.INCOMPLETE
        self.questions: str
        self.eval_: str = ""
        self.best = {"report": "", "score": 0}
    
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

        prompt = SYSTEM_PROMPT + f"\nTopic document:\n{self.topic}\n \nReport:\n{report}\nIR Context:\n{ir_context or 'First round no IR context yet'}\nGenerator Comments:\n{self.serialize_notes(False)}\n Your Comments:\n{self.serialize_notes(True)}"
        self._log(prompt)
        response: Response = await gated_call_gen(
            prompt = prompt,
            client=self.client,
            temperature=0.2,
        )
        text = response.output_text
        if not text or not text.strip():
            raise ValueError("Empty response from LLM")
        self._log(text)
        self._update_status(text,report)
        return self.my_notes[-1],self.questions, self.eval_
    
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
    
    @staticmethod
    def _ensure_file(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: f.write("")

    def _update_status(self, content,report):
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
        try:
            note,self.questions = self._extract_tag(content, "note"), json.loads(self._extract_tag(content, "ir")).get("questions", [])
            self.my_notes.append(note)
            self._eval = self._extract_tag(content, "eval")
            self._update_eval(json.loads(self._eval),report)
        except Exception as e:
            print(e)
            self.questions = []
            self.status = EvalStatus.FAIL
            self.my_notes.append("Error parsing evaluation")

    def _update_eval(self,eval_: Dict[str, int],report: str) -> None:
        total = 0
        MAX = 15 + 10 + 10 + 5 + 5 + 10 + 10
        for key,val in eval_.items():
            match key:
                case "coverage":
                    # max 15
                    total += 3 * val
                    
                case "accuracy":
                    # max 10 
                    total += 2 * val
                    
                case "citation_quality":
                    # max 10 
                    total += 2 * val
                    
                case "style":
                    # max 5
                    total += val
                    
                case "prioritization":
                    # max 5
                    total += val
                    
                case "completeness":
                    # max 10
                    total += val
                case "information_density":
                    # max 10
                    total += val
                case _:
                    pass
        print(f"Current eval score: {total}/{MAX}")
        if total >= self.best['score'] :
            self.best['score'] = total
            self.best['report'] = report
        self.status = EvalStatus.PASS if total/MAX >= .9 else EvalStatus.FAIL