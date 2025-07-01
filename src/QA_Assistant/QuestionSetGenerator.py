"""
Question Set Generator - Combines Template Generator and Summary Extractor
Creates a unified pipeline that generates questions from documents and analyzes which can be answered from the document.
"""

import json
import asyncio
import requests
from typing import List, Dict, Any
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QA_Assistant.SummaryExtractor import SummaryExtractor

class QuestionSetGenerator:
    def __init__(self, langflow_url: str = "http://127.0.0.1:7860/api/v1/run/d291fdb7-25cf-463d-a43e-a1a08e2ace49"):
        self.langflow_url = langflow_url
        self.summary_extractor = SummaryExtractor()
        
    def _combine_group_json_texts(self, json_texts: List[str], indent: int = 2) -> str:
        """Combine multiple JSON responses from the template generator."""
        merged = {"groups": []}
        
        for jtxt in json_texts:
            try:
                data = json.loads(jtxt)
                merged["groups"].extend(data.get("groups", []))
            except json.JSONDecodeError:
                continue
                
        return json.dumps(merged, ensure_ascii=False, indent=indent)
    
    def _extract_questions_from_groups(self, groups_json: str) -> List[str]:
        """Extract all questions from the grouped JSON response."""
        try:
            data = json.loads(groups_json)
            questions = []
            
            for group in data.get("groups", []):
                questions.extend(group.get("questions", []))
                
            return questions
        except json.JSONDecodeError:
            return []
    
    async def _generate_questions_from_document(self, document_text: str) -> List[str]:
        """Generate questions using the template generator via Langflow API."""
        payload = {
            "input_value": document_text,
            "output_type": "text",
            "input_type": "text"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.langflow_url, json=payload, headers=headers)
            response.raise_for_status()
            
            response_data = json.loads(response.text)
            
            # Extract the three outputs from the Langflow response
            outputs = [
                response_data["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"],
                response_data["outputs"][0]["outputs"][1]["results"]["text"]["data"]["text"],
                response_data["outputs"][0]["outputs"][2]["results"]["text"]["data"]["text"]
            ]
            
            # Combine the outputs
            combined_json = self._combine_group_json_texts(outputs)
            
            # Extract questions from the combined JSON
            questions = self._extract_questions_from_groups(combined_json)
            
            return questions
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request to template generator: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing template generator response: {e}")
            return []
    
    async def get_question_set(self, document: str) -> Dict[str, Any]:
        """
        Generate questions from a document and analyze which can be answered from the document.
        
        Args:
            document: The document text to analyze
            
        Returns:
            Dict containing the document, generated questions, and analysis
        """
        try:
            # Step 1: Generate questions using the template generator
            print("Generating questions from document...")
            questions = await self._generate_questions_from_document(document)
            
            if not questions:
                return {
                    "error": "Failed to generate questions from document",
                    "document": document[:200] + "..." if len(document) > 200 else document
                }
            
            print(f"Generated {len(questions)} questions")
            
            # Step 2: Analyze which questions can be answered from the document
            print("Analyzing questions for document-based answers...")
            analysis = await self.summary_extractor.extract_question_contexts(document, questions)
            
            # Step 3: Return the combined result
            result = {
                "document": document,
                "generated_questions": questions,
                "analysis": analysis,
                "metadata": {
                    "total_questions": len(questions),
                    "questions_answered_from_doc": analysis.get("metadata", {}).get("answered_count", 0),
                    "questions_needing_external_info": analysis.get("metadata", {}).get("unanswered_count", 0)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error in get_question_set: {e}",
                "document": document[:200] + "..." if len(document) > 200 else document
            }

async def main():
    """Test function for the QuestionSetGenerator"""
    generator = QuestionSetGenerator()
    
    # Sample document for testing
    sample_document = """
    Artificial Intelligence in Healthcare: A Comprehensive Overview
    
    Artificial Intelligence (AI) has revolutionized healthcare delivery in recent years. 
    Machine learning algorithms can now diagnose diseases with accuracy rates exceeding 95% 
    in some cases, particularly in radiology and pathology. Deep learning models trained 
    on millions of medical images can detect early signs of cancer, heart disease, and 
    neurological disorders.
    
    The implementation of AI in hospitals has led to a 30% reduction in diagnostic errors 
    and a 25% improvement in patient outcomes. However, challenges remain, including 
    concerns about data privacy, algorithmic bias, and the need for regulatory oversight.
    
    Key applications include:
    - Medical imaging analysis
    - Drug discovery and development
    - Patient monitoring and predictive analytics
    - Administrative task automation
    
    The global AI healthcare market is expected to reach $45.2 billion by 2026, driven 
    by increasing demand for personalized medicine and cost-effective healthcare solutions.
    """
    
    print("Testing QuestionSetGenerator...")
    result = await generator.get_question_set(sample_document)
    
    print("\n" + "="*60)
    print("QUESTION SET GENERATOR RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 