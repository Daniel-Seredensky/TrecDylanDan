"""
Enhanced Summary Extractor for Question-Specific Context
Extracts relevant document context for each question with credibility assessment.
"""

import json
import asyncio
from typing import List, Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class SummaryExtractor:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        
    async def extract_question_contexts(
        self, 
        topic_document: str, 
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Extract relevant context from topic document for each question with credibility assessment.
        
        Args:
            topic_document: The full topic document text
            questions: List of questions to extract context for
            
        Returns:
            Dict with answered/unanswered questions and credibility analysis
        """
        
        system_prompt = """You are Article-Mediator/Credibility-Agent v2.0, a specialized reasoning agent that analyzes documents and determines which questions can be answered from the document alone versus those requiring external information.

Your mission is to:
1. Answer questions that can be fully resolved using only the document
2. For questions requiring external knowledge, provide relevant document context
3. Assess credibility signals (author tone, bias, rhetorical devices, source context)
4. Minimize token usage by extracting only essential information

For each question, analyze:
- Can this be answered completely from the document?
- What specific passages are relevant?
- What credibility signals should be noted?
- What external information would be needed (if any)?

Output format:
{
  "answered": [
    {
      "question": "<verbatim question>",
      "answer": "<complete answer based solely on document>"
    }
  ],
  "unanswered": [
    {
      "question": "<verbatim question>",
      "doc_context": "<relevant excerpts + credibility notes + what needs external info>"
    }
  ]
}

Guidelines:
- Keep answers concise (≤90 words)
- Include credibility signals (tone, bias, author background)
- Use short excerpts (≤30 words) as evidence
- Be objective and cite the text
- Mark questions as "unanswered" if they require any external knowledge"""

        user_prompt = f"""DOCUMENT:
{topic_document}

QUESTIONS:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(questions))}

Analyze each question and provide the appropriate response format. Focus on credibility assessment including author tone, potential bias, rhetorical devices, and source context."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content
            if result_text is None:
                return {
                    "error": "Empty response from API",
                    "questions": questions
                }
            result_text = result_text.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                
                # Validate structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure required keys exist
                if "answered" not in result:
                    result["answered"] = []
                if "unanswered" not in result:
                    result["unanswered"] = []
                
                # Add metadata
                result["metadata"] = {
                    "total_questions": len(questions),
                    "answered_count": len(result["answered"]),
                    "unanswered_count": len(result["unanswered"]),
                    "questions": questions
                }
                
                return result
                
            except json.JSONDecodeError as e:
                return {
                    "error": f"Failed to parse JSON response: {e}",
                    "raw_response": result_text,
                    "questions": questions
                }
                
        except Exception as e:
            return {
                "error": f"API call failed: {e}",
                "questions": questions
            }
    
    def get_context_for_question(self, result: Dict[str, Any], question_index: int) -> str:
        """
        Extract context for a specific question from the result.
        
        Args:
            result: Result from extract_question_contexts
            question_index: Index of the question (0-based)
            
        Returns:
            The context string for that question
        """
        if "error" in result:
            return f"Error: {result['error']}"
            
        question = result["metadata"]["questions"][question_index]
        
        # Check answered questions first
        for answered in result.get("answered", []):
            if answered["question"] == question:
                return answered["answer"]
        
        # Check unanswered questions
        for unanswered in result.get("unanswered", []):
            if unanswered["question"] == question:
                return unanswered["doc_context"]
        
        return "Question not found in results"

async def main():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from QA_Assistant.Searcher import search
    extractor = SummaryExtractor()
    print("Searching Marco index for a political article...")
    queries = ["political election campaign", "government policy legislation"]
    master_query = "political elections and government policy in modern democracy"
    agent_id = "summary_test"
    try:
        search_results = await search(queries, master_query, agent_id)
        if not search_results:
            raise RuntimeError("No political documents found in Marco index")
        first_result = search_results[0]
        sample_doc = f"""
        Title: {first_result.get('title', 'No title')}
        URL: {first_result.get('url', 'No URL')}
        Headings: {first_result.get('headings', 'No headings')}
        Segment ID: {first_result.get('segment_id', 'No ID')}
        This is a political document from the Marco index. In a real implementation, 
        you would fetch the full document content using the segment_id or URL.
        """
        questions = [
            "What is the main topic of this document?",
            "What are the key issues discussed?",
            "What is the capital of France?",
            "What statistics or data are mentioned?",
            "What trends or patterns are described?"
        ]
        result = await extractor.extract_question_contexts(sample_doc, questions)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 