import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import sys
import importlib.util
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Dynamically import template_generator
TEMPLATE_GEN_PATH = Path(__file__).resolve().parent.parent / 'DebateAndReport' / 'template_generator' / 'template_generator.py'
spec = importlib.util.spec_from_file_location("template_generator", str(TEMPLATE_GEN_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load template_generator from {TEMPLATE_GEN_PATH}")
template_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template_generator)

import openai

class AsyncQuestionSetGenerator:
    def __init__(self, document: str, question_template: Dict[str, Any], model: str = "gpt-4o", batch_size: int = 10, max_concurrent: int = 2, groups_per_call: int = 2):
        self.document = document
        self.question_template = question_template
        self.model = model or os.getenv("OPENAI_QUESTIONSET_MODEL", "gpt-4o")
        self.client = openai.AsyncOpenAI()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.groups_per_call = groups_per_call
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

    async def process_groups(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Combine all questions from the given groups
        all_questions = []
        for group in groups:
            all_questions.extend(group.get("questions", []))
        answered = []
        unanswered = []
        for i in range(0, len(all_questions), self.batch_size):
            batch = all_questions[i:i+self.batch_size]
            prompt = self._build_prompt(batch)
            result = await self._call_openai(prompt)
            if "error" in result:
                return result
            answered.extend(result.get("answered", []))
            unanswered.extend(result.get("unanswered", []))
        return {"answered": answered, "unanswered": unanswered}

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
        # Chunk groups into groups_per_call
        group_chunks = [groups[i:i+self.groups_per_call] for i in range(0, len(groups), self.groups_per_call)]
        tasks = [self.process_groups(chunk) for chunk in group_chunks if chunk]
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

def run_question_set_pipeline(document_text: str, questions: Optional[List[str]] = None, output_path: Optional[str] = None, model: str = "gpt-4o", batch_size: int = 10, max_concurrent: int = 2) -> dict:
    """
    Runs the question set pipeline with raw document text and optional questions.
    If questions are not provided, they will be generated automatically from the document.
    
    Args:
        document_text: Raw text of the document to analyze
        questions: Optional list of questions to ask about the document
        output_path: Optional path to save the result
        model: OpenAI model to use
        batch_size: Number of questions to process in each batch
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        Dictionary with answered and unanswered questions
    """
    async def _run():
        # If no questions provided, generate them from the document
        if questions is None:
            # Generate questions using a simple prompt
            question_generation_prompt = f"""
            Based on the following document, generate 10 relevant questions that would help 
            understand the key points, implications, and details of the content.
            
            Document:
            {document_text}
            
            Generate questions that cover:
            - Main topics and themes
            - Key findings or conclusions
            - Implications and consequences
            - Technical details and processes
            - Comparisons and contrasts mentioned
            
            Return only the questions, one per line, without numbering or bullet points.
            """
            
            # Generate questions using OpenAI
            client = openai.AsyncOpenAI()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates relevant questions about documents."},
                        {"role": "user", "content": question_generation_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                generated_questions_text = response.choices[0].message.content
                if generated_questions_text:
                    # Split into individual questions and clean up
                    generated_questions = [q.strip() for q in generated_questions_text.split('\n') if q.strip()]
                    questions_to_use = generated_questions[:10]  # Limit to 10 questions
                else:
                    # Fallback questions if generation fails
                    questions_to_use = [
                        "What is the main topic of this document?",
                        "What are the key points discussed?",
                        "What are the main conclusions?",
                        "What implications does this have?",
                        "What are the main challenges mentioned?",
                        "What are the benefits described?",
                        "What processes or methods are discussed?",
                        "What comparisons are made?",
                        "What recommendations are given?",
                        "What future considerations are mentioned?"
                    ]
            finally:
                await client.close()
        else:
            questions_to_use = questions
        
        # Create a simple question template structure
        question_template = {
            "groups": [
                {
                    "questions": questions_to_use
                }
            ]
        }
        
        generator = AsyncQuestionSetGenerator(document_text, question_template, model=model, batch_size=batch_size, max_concurrent=max_concurrent)
        try:
            result = await generator.generate()
        finally:
            await generator.client.close()
            
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with output_path_obj.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    return asyncio.run(_run())

def run_question_set_with_doc_id(doc_id: str, output_path: Optional[str] = None) -> dict:
    """
    Runs the full question set pipeline for a given doc_id.
    If output_path is provided, writes the result to that path.
    Returns the result as a dict.
    """
    async def _run():
        question_template_json = template_generator.create_question_template(doc_id)
        if not question_template_json:
            raise RuntimeError("Failed to generate question template.")
        question_template = json.loads(question_template_json)
        document = template_generator.get_document_text(doc_id, "DebateAndReport/FixedSampleData.jsonl")
        generator = AsyncQuestionSetGenerator(document, question_template, max_concurrent=2)
        try:
            result = await generator.generate()
        finally:
            await generator.client.close()
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with output_path_obj.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    return asyncio.run(_run())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        doc_id = sys.argv[1]
    else:
        doc_id = "msmarco_v2.1_doc_17_795452723#14_866362073"  # Example doc id
    # Default output to DerivedData/QuestionSets/{doc_id}.json
    safe_doc_id = doc_id.replace('/', '_').replace('#', '_')
    output_path = f"DerivedData/QuestionSets/{safe_doc_id}.json"
    result = run_question_set_with_doc_id(doc_id, output_path=output_path)
    print(f"Wrote question set output to {output_path}")
