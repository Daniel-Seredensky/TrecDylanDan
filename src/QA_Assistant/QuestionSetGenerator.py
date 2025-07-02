import json
from pathlib import Path
from typing import List, Dict, Any
import os
import re
import sys
import importlib.util
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Dynamically import template_generator
TEMPLATE_GEN_PATH = Path(__file__).resolve().parent.parent.parent / 'DebateAndReport' / 'template_generator' / 'template_generator.py'
spec = importlib.util.spec_from_file_location("template_generator", str(TEMPLATE_GEN_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load template_generator from {TEMPLATE_GEN_PATH}")
template_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template_generator)

import openai

class AsyncQuestionSetGenerator:
    def __init__(self, document: str, question_template: Dict[str, Any], model: str = "gpt-4o", batch_size: int = 10, max_concurrent: int = 2):
        self.document = document
        self.question_template = question_template
        self.model = model or os.getenv("OPENAI_QUESTIONSET_MODEL", "gpt-4o")
        self.client = openai.AsyncOpenAI()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _build_prompt(self, questions: List[str]) -> str:
        joined_questions = "\n".join(f"- {q}" for q in questions)
        return (
            f"DOCUMENT:\n{self.document}\n\nQUESTIONS:\n"
            f"{joined_questions}\n\n"
            "For each question, answer strictly in the following JSON format (no markdown, no commentary, no comments, no trailing commas, no extra text):\n"
            "{\n"
            "  \"answered\": [\n"
            "    {\n"
            "      \"question\": \"<verbatim question>\",\n"
            "      \"answer\": \"<full answer based solely on the document>\"\n"
            "    }\n"
            "  ],\n"
            "  \"unanswered\": [\n"
            "    {\n"
            "      \"question\": \"<verbatim question>\",\n"
            "      \"doc_context\": \"<succinct excerpts and/or summary + credibility notes that explain what's known and what still needs external info>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Omit any array that would be empty. Do not add extra keys, commentary, or top-level text outside this schema. Output only valid JSON."
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object boundaries
            start = text.find('{')
            if start == -1:
                return {"error": "No JSON object found in response.", "raw_response": text}
            
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if brace_count != 0:
                # Try simple fixes for common JSON issues
                text_fixed = re.sub(r',\s*([}\]])', r'\1', text)
                try:
                    return json.loads(text_fixed)
                except Exception as e:
                    return {"error": f"Failed to parse JSON after fixes: {e}", "raw_response": text}
            
            try:
                json_str = text[start:end]
                return json.loads(json_str)
            except Exception as e:
                return {"error": f"Failed to parse extracted JSON: {e}", "raw_response": text}
        except Exception as e:
            return {"error": f"Unexpected error during JSON extraction: {e}", "raw_response": text}

    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                await asyncio.sleep(1)  # Rate limiting delay
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are Article-Mediator/Credibility-Agent v2.0. Follow the instructions exactly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4096
                )
                result_text = response.choices[0].message.content
                if not result_text:
                    return {"error": "Empty response from OpenAI."}
                result = self._extract_json(result_text)
                if not isinstance(result, dict):
                    return {"error": "Response is not a dictionary", "raw_response": result_text}
                if "answered" not in result and "unanswered" not in result:
                    return {"error": "Missing both 'answered' and 'unanswered' keys", "raw_response": result_text}
                return result
            except openai.RateLimitError as e:
                return {"error": f"Rate limit exceeded: {e}"}
            except Exception as e:
                return {"error": f"OpenAI API call failed: {e}"}

    async def process_group(self, questions: List[str]) -> Dict[str, Any]:
        answered = []
        unanswered = []
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i+self.batch_size]
            prompt = self._build_prompt(batch)
            result = await self._call_openai(prompt)
            if "error" in result:
                return result
            answered.extend(result.get("answered", []))
            unanswered.extend(result.get("unanswered", []))
        return {"answered": answered, "unanswered": unanswered}

    async def generate(self) -> Dict[str, Any]:
        groups = self.question_template.get("groups", [])
        tasks = [self.process_group(group.get("questions", [])) for group in groups if group.get("questions")]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        answered = []
        unanswered = []
        for result in results:
            if isinstance(result, Exception):
                return {"error": f"Task failed with exception: {result}"}
            if isinstance(result, dict) and "error" in result:
                return result
            if isinstance(result, dict):
                answered.extend(result.get("answered", []))
                unanswered.extend(result.get("unanswered", []))
        output = {}
        if answered:
            output["answered"] = answered
        if unanswered:
            output["unanswered"] = unanswered
        return output

if __name__ == "__main__":
    async def main():
        if len(sys.argv) > 1:
            doc_id = sys.argv[1]
        else:
            doc_id = "msmarco_v2.1_doc_17_795452723#14_866362073"  # Example doc id
        print(f"Generating question template for doc_id: {doc_id}")
        question_template_json = template_generator.create_question_template(doc_id)
        if not question_template_json:
            print("Failed to generate question template.")
            sys.exit(1)
        question_template = json.loads(question_template_json)
        document = template_generator.get_document_text(doc_id, "DebateAndReport/FixedSampleData.jsonl")
        generator = AsyncQuestionSetGenerator(document, question_template, max_concurrent=2)
        output = await generator.generate()
        output_path = Path("DebateAndReport/output/question_set_output.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Wrote question set output to {output_path}")
    asyncio.run(main()) 