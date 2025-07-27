#!/usr/bin/env python3
"""
Convert current pipeline output to TREC-2025-DRAGUN run format.

This script takes the current pipeline output (like japan_report.json) and
converts it to the required TREC run format.
"""

import json
import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path


def convert_pipeline_output_to_run(pipeline_output: Dict[str, Any], 
                                 topic_id: str, team_id: str = "SCIAI",
                                 run_id: str = "SCIAI-run-example",
                                 run_type: str = "automatic",
                                 use_starter_kit: int = 0) -> Dict[str, Any]:
    """
    Convert pipeline output to TREC run format.
    
    Args:
        pipeline_output: Current pipeline output (like japan_report.json)
        topic_id: The topic document ID
        team_id: Team identifier
        run_id: Run identifier
        run_type: Type of run ("automatic" or "manual")
        use_starter_kit: Whether the run uses the starter kit (0 or 1)
        
    Returns:
        Run entry in the required format
    """
    # Extract responses from pipeline output
    responses = pipeline_output.get('responses', [])
    
    # Validate and clean responses
    cleaned_responses = []
    total_words = 0
    
    for response in responses:
        if 'text' not in response:
            continue
            
        text = response['text'].strip()
        if not text:
            continue
            
        # Count words
        word_count = len(text.split())
        total_words += word_count
        
        # Get citations (ensure they're in the right format)
        citations = response.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        # Limit to 3 citations
        citations = citations[:3]
        
        cleaned_response = {
            "text": text,
            "citations": citations
        }
        cleaned_responses.append(cleaned_response)
    
    # Check word limit
    if total_words > 250:
        print(f"Warning: Total word count ({total_words}) exceeds 250 word limit", file=sys.stderr)
        # You might want to truncate or split responses here
    
    # Create run entry
    run_entry = {
        "metadata": {
            "team_id": team_id,
            "run_id": run_id,
            "topic_id": topic_id,
            "type": run_type,
            "use_starter_kit": use_starter_kit
        },
        "responses": cleaned_responses
    }
    
    return run_entry


def convert_file(input_file: str, output_file: str, topic_id: str,
                team_id: str = "SCIAI", run_id: str = "SCIAI-run-example",
                run_type: str = "automatic", use_starter_kit: int = 0):
    """
    Convert a pipeline output file to run format.
    
    Args:
        input_file: Path to the pipeline output file
        output_file: Path to the output run file
        topic_id: The topic document ID
        team_id: Team identifier
        run_id: Run identifier
        run_type: Type of run ("automatic" or "manual")
        use_starter_kit: Whether the run uses the starter kit (0 or 1)
    """
    try:
        # Load pipeline output
        with open(input_file, 'r', encoding='utf-8') as f:
            pipeline_output = json.load(f)
        
        # Convert to run format
        run_entry = convert_pipeline_output_to_run(
            pipeline_output=pipeline_output,
            topic_id=topic_id,
            team_id=team_id,
            run_id=run_id,
            run_type=run_type,
            use_starter_kit=use_starter_kit
        )
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(run_entry, f, ensure_ascii=False, indent=2)
        
        print(f"Converted {input_file} to {output_file}")
        print(f"Topic ID: {topic_id}")
        print(f"Responses: {len(run_entry['responses'])}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert pipeline output to TREC-2025-DRAGUN run format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_run_format.py japan_report.json japan_run.json msmarco_v2.1_doc_xx_xxxxx
  python convert_to_run_format.py report.json run.json topic_id --team-id SCIAI --run-id SCIAI-run-1
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the pipeline output file (e.g., japan_report.json)'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to the output run file'
    )
    
    parser.add_argument(
        'topic_id',
        help='The topic document ID (e.g., msmarco_v2.1_doc_04_420132660)'
    )
    
    parser.add_argument(
        '--team-id',
        default='SCIAI',
        help='Team identifier (default: SCIAI)'
    )
    
    parser.add_argument(
        '--run-id',
        default='SCIAI-run-example',
        help='Run identifier (default: SCIAI-run-example)'
    )
    
    parser.add_argument(
        '--type',
        choices=['automatic', 'manual'],
        default='automatic',
        help='Type of run (default: automatic)'
    )
    
    parser.add_argument(
        '--use-starter-kit',
        type=int,
        choices=[0, 1],
        default=0,
        help='Whether the run uses the starter kit (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Convert file
    convert_file(
        input_file=args.input_file,
        output_file=args.output_file,
        topic_id=args.topic_id,
        team_id=args.team_id,
        run_id=args.run_id,
        run_type=args.type,
        use_starter_kit=args.use_starter_kit
    )


if __name__ == "__main__":
    main() 