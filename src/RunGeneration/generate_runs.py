#!/usr/bin/env python3
"""
Generate runs from TREC-2025-DRAGUN topics file.

This script processes the trec-2025-dragun-topics.jsonl file and generates
runs in the required format for the TREC-2025-DRAGUN evaluation.
"""

import json
import sys
import asyncio
import uvloop
from typing import List, Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Import your existing pipeline components
from openai import AsyncAzureOpenAI, AsyncOpenAI
from ..IR_Ensemble.QA_Assistant.bucket_monitor import BucketMonitor
from ..IR_Ensemble.QA_Assistant.Searcher import cohere_client
from ..IR_Ensemble.QA_Assistant.daemon_wrapper import JVMDaemon 
from ..IR_Ensemble.context_builder import ContextProctor
from ..ReportGenerator.report_generator import ReportGenerator
from ..ReportEvaluator.report_evaluator import ReportEvaluator, EvalStatus

load_dotenv()
MAX_ROUNDS = 3


def load_topics(topics_file: str) -> List[Dict[str, Any]]:
    """
    Load topics from the JSONL file.
    
    Args:
        topics_file: Path to the topics JSONL file
        
    Returns:
        List of topic dictionaries
    """
    topics = []
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    topic = json.loads(line)
                    topics.append(topic)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: Topics file '{topics_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading topics file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return topics


async def get_context(client: AsyncAzureOpenAI, questions: list[dict[str,str]]) -> str:
    """
    Get search results for a set of questions.
    """
    proc = ContextProctor(client, questions)
    try: 
        await proc.create_context()
        # clear mem
        del proc
        with open(os.getenv("CONTEXT_PATH"), "r") as f: 
            return f.read()
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        del proc
        return "Error generating context."


async def generate_report_for_topic(oai_client: AsyncOpenAI, aoai_client: AsyncAzureOpenAI, topic: Dict[str, Any]) -> str:
    """
    Generate a report for a specific topic using the existing pipeline.
    
    Args:
        oai_client: OpenAI client
        aoai_client: Azure OpenAI client
        topic: Topic dictionary
        
    Returns:
        Generated report text
    """
    # Extract topic content
    topic_content = f"Title: {topic.get('title', '')}\n\nBody: {topic.get('body', '')}"
    
    # clear old context if it exists else ensure it exists
    with open(os.getenv("CONTEXT_PATH"), "w") as f: 
        f.write("")

    # init agents
    gen: ReportGenerator = ReportGenerator(client=oai_client, topic=topic_content)
    eval: ReportEvaluator = ReportEvaluator(client=oai_client, topic=topic_content)
    rounds = 0
    note, context = [None]*2

    # run loop
    while rounds < MAX_ROUNDS:
        report, note = await gen.generate_report(context, note)
        note, questions = await eval.evaluate(report=report, generator_comment=note, ir_context=context)
        if eval.status == EvalStatus.PASS:
            break
        context = await get_context(client=aoai_client, questions=questions)
        rounds += 1
    
    return report


def split_report_into_responses(report: str, topic_docid: str) -> List[Dict[str, Any]]:
    """
    Split a report into individual responses with citations.
    
    Args:
        report: The generated report text
        topic_docid: The document ID for citations
        
    Returns:
        List of response dictionaries
    """
    # Split report into sentences (simple approach)
    sentences = []
    current_sentence = ""
    
    for char in report:
        current_sentence += char
        if char in '.!?':
            sentence = current_sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short sentences
                sentences.append(sentence)
            current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Convert to response format
    responses = []
    for i, sentence in enumerate(sentences):
        # Create sample citations (you should replace this with actual citation logic)
        citations = []
        if i < 3:  # Add citations to first few sentences
            citations = [f"{topic_docid}#{i}_{hash(sentence) % 1000000000}"]
        
        response = {
            "text": sentence,
            "citations": citations
        }
        responses.append(response)
    
    return responses


def create_run_entry(topic: Dict[str, Any], responses: List[Dict[str, Any]], 
                    team_id: str, run_id: str, run_type: str = "automatic", 
                    use_starter_kit: int = 0) -> Dict[str, Any]:
    """
    Create a run entry in the required format.
    
    Args:
        topic: Topic dictionary
        responses: List of response dictionaries
        team_id: Team identifier
        run_id: Run identifier
        run_type: Type of run ("automatic" or "manual")
        use_starter_kit: Whether the run uses the starter kit (0 or 1)
        
    Returns:
        Run entry dictionary in the required format
    """
    return {
        "metadata": {
            "team_id": team_id,
            "run_id": run_id,
            "topic_id": topic.get('docid', 'unknown'),
            "type": run_type,
            "use_starter_kit": use_starter_kit
        },
        "responses": responses
    }


def validate_responses(responses: List[Dict[str, Any]]) -> bool:
    """
    Validate that responses meet the requirements.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        True if valid, False otherwise
    """
    total_words = 0
    
    for response in responses:
        if 'text' not in response:
            print("Error: Response missing 'text' field", file=sys.stderr)
            return False
        
        if 'citations' not in response:
            print("Error: Response missing 'citations' field", file=sys.stderr)
            return False
        
        # Count words in text
        text = response['text']
        word_count = len(text.split())
        total_words += word_count
        
        # Check citations format
        citations = response['citations']
        if not isinstance(citations, list):
            print("Error: Citations must be a list", file=sys.stderr)
            return False
        
        if len(citations) > 3:
            print("Error: Maximum 3 citations allowed per response", file=sys.stderr)
            return False
        
        for citation in citations:
            if not isinstance(citation, str):
                print("Error: Citations must be strings", file=sys.stderr)
                return False
    
    if total_words > 250:
        print(f"Error: Total word count ({total_words}) exceeds 250 word limit", file=sys.stderr)
        return False
    
    return True


async def generate_runs_async(topics_file: str, output_file: str, team_id: str = "SCIAI", 
                             run_id: str = "SCIAI-run-example", run_type: str = "automatic",
                             use_starter_kit: int = 0, max_topics: int = None):
    """
    Generate runs from topics file using the async pipeline.
    
    Args:
        topics_file: Path to the topics JSONL file
        output_file: Path to the output JSONL file
        team_id: Team identifier
        run_id: Run identifier
        run_type: Type of run ("automatic" or "manual")
        use_starter_kit: Whether the run uses the starter kit (0 or 1)
        max_topics: Maximum number of topics to process (for testing)
    """
    # Load topics
    topics = load_topics(topics_file)
    if max_topics:
        topics = topics[:max_topics]
    print(f"Loaded {len(topics)} topics from {topics_file}")
    
    # Initialize clients
    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=25,
        max_retries=3,
    )
    aoai_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="preview",
        timeout=60.5,
        max_retries=3,
    )
    
    # Initialize bucket monitor
    bm = BucketMonitor()
    
    try:
        await bm.start()
        
        # Generate runs
        runs = []
        for i, topic in enumerate(topics, 1):
            print(f"Processing topic {i}/{len(topics)}: {topic.get('docid', 'unknown')}")
            
            try:
                # Generate report for this topic
                report = await generate_report_for_topic(oai_client, aoai_client, topic)
                
                # Split report into responses
                responses = split_report_into_responses(report, topic.get('docid', 'unknown'))
                
                # Validate responses
                if not validate_responses(responses):
                    print(f"Error: Invalid responses for topic {topic.get('docid', 'unknown')}", file=sys.stderr)
                    continue
                
                # Create run entry
                run_entry = create_run_entry(
                    topic=topic,
                    responses=responses,
                    team_id=team_id,
                    run_id=run_id,
                    run_type=run_type,
                    use_starter_kit=use_starter_kit
                )
                
                runs.append(run_entry)
                print(f"Successfully generated run for topic {topic.get('docid', 'unknown')}")
                
            except Exception as e:
                print(f"Error processing topic {topic.get('docid', 'unknown')}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue
        
        # Write output
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for run in runs:
                    json.dump(run, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Generated {len(runs)} runs and saved to {output_file}")
            
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error in pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        await oai_client.close()
        await aoai_client.close()
        await cohere_client.aclose()
        await JVMDaemon.stop()
        await bm.stop()


def generate_runs(topics_file: str, output_file: str, team_id: str = "SCIAI", 
                 run_id: str = "SCIAI-run-example", run_type: str = "automatic",
                 use_starter_kit: int = 0, max_topics: int = None):
    """
    Wrapper function to run the async generation.
    """
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(generate_runs_async(
        topics_file=topics_file,
        output_file=output_file,
        team_id=team_id,
        run_id=run_id,
        run_type=run_type,
        use_starter_kit=use_starter_kit,
        max_topics=max_topics
    ))


def main():
    """Main function with hard-coded parameters."""
    # Hard-coded parameters
    topics_file = "trec-2025-dragun-topics.jsonl"
    output_file = "runs.jsonl"
    team_id = "SCIAI"
    run_id = "SCIAI-run-example"
    run_type = "automatic"
    use_starter_kit = 0
    max_topics = 3  # Set to a number for testing, e.g., 5
    
    # Validate input file exists
    if not Path(topics_file).exists():
        print(f"Error: Topics file '{topics_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Generate runs
    generate_runs(
        topics_file=topics_file,
        output_file=output_file,
        team_id=team_id,
        run_id=run_id,
        run_type=run_type,
        use_starter_kit=use_starter_kit,
        max_topics=max_topics
    )


if __name__ == "__main__":
    main() 