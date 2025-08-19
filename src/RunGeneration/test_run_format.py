#!/usr/bin/env python3
"""
Test script to verify run format compliance.

This script validates that generated runs follow the TREC-2025-DRAGUN format
requirements.
"""

import json
import sys
from typing import List, Dict, Any


def validate_run_format(run: Dict[str, Any]) -> List[str]:
    """
    Validate a single run entry.
    
    Args:
        run: Run dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    if 'metadata' not in run:
        errors.append("Missing 'metadata' field")
        return errors
    
    if 'responses' not in run:
        errors.append("Missing 'responses' field")
        return errors
    
    # Validate metadata
    metadata = run['metadata']
    required_metadata_fields = ['team_id', 'run_id', 'topic_id', 'type', 'use_starter_kit']
    
    for field in required_metadata_fields:
        if field not in metadata:
            errors.append(f"Missing metadata field: {field}")
    
    # Validate metadata field types
    if 'team_id' in metadata and not isinstance(metadata['team_id'], str):
        errors.append("team_id must be a string")
    
    if 'run_id' in metadata and not isinstance(metadata['run_id'], str):
        errors.append("run_id must be a string")
    
    if 'topic_id' in metadata and not isinstance(metadata['topic_id'], str):
        errors.append("topic_id must be a string")
    
    if 'type' in metadata and metadata['type'] not in ['automatic', 'manual']:
        errors.append("type must be 'automatic' or 'manual'")
    
    if 'use_starter_kit' in metadata and metadata['use_starter_kit'] not in [0, 1]:
        errors.append("use_starter_kit must be 0 or 1")
    
    # Validate responses
    responses = run['responses']
    if not isinstance(responses, list):
        errors.append("responses must be a list")
        return errors
    
    if not responses:
        errors.append("responses list cannot be empty")
        return errors
    
    # Validate each response
    total_words = 0
    for i, response in enumerate(responses):
        if not isinstance(response, dict):
            errors.append(f"Response {i} must be a dictionary")
            continue
        
        # Check required response fields
        if 'text' not in response:
            errors.append(f"Response {i} missing 'text' field")
        elif not isinstance(response['text'], str):
            errors.append(f"Response {i} text must be a string")
        else:
            # Count words
            word_count = len(response['text'].split())
            total_words += word_count
        
        if 'citations' not in response:
            errors.append(f"Response {i} missing 'citations' field")
        elif not isinstance(response['citations'], list):
            errors.append(f"Response {i} citations must be a list")
        else:
            # Validate citations
            if len(response['citations']) > 3:
                errors.append(f"Response {i} has more than 3 citations")
            
            for j, citation in enumerate(response['citations']):
                if not isinstance(citation, str):
                    errors.append(f"Response {i} citation {j} must be a string")
    
    # Check word limit
    if total_words > 250:
        errors.append(f"Total word count ({total_words}) exceeds 250 word limit")
    
    return errors


def validate_runs_file(runs_file: str) -> bool:
    """
    Validate all runs in a JSONL file or single JSON file.
    
    Args:
        runs_file: Path to the file containing runs (JSONL or single JSON)
        
    Returns:
        True if all runs are valid, False otherwise
    """
    try:
        with open(runs_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as single JSON first
        try:
            run = json.loads(content)
            errors = validate_run_format(run)
            
            if errors:
                print(f"Single JSON file - Errors:")
                for error in errors:
                    print(f"  - {error}")
                return False
            else:
                topic_id = run.get('metadata', {}).get('topic_id', 'unknown')
                response_count = len(run.get('responses', []))
                print(f"Single JSON file - Topic {topic_id}: {response_count} responses - VALID")
                print("\n✅ Run is valid!")
                return True
                
        except json.JSONDecodeError:
            # If single JSON fails, try as JSONL
            lines = content.split('\n')
            print(f"Validating {len(lines)} runs from {runs_file} (JSONL format)")
            
            all_valid = True
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    run = json.loads(line)
                    errors = validate_run_format(run)
                    
                    if errors:
                        print(f"Line {line_num} - Errors:")
                        for error in errors:
                            print(f"  - {error}")
                        all_valid = False
                    else:
                        topic_id = run.get('metadata', {}).get('topic_id', 'unknown')
                        response_count = len(run.get('responses', []))
                        print(f"Line {line_num} - Topic {topic_id}: {response_count} responses - VALID")
                        
                except json.JSONDecodeError as e:
                    print(f"Line {line_num} - JSON decode error: {e}")
                    all_valid = False
            
            if all_valid:
                print("\n✅ All runs are valid!")
            else:
                print("\n❌ Some runs have validation errors.")
            
            return all_valid
        
    except FileNotFoundError:
        print(f"Error: File '{runs_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
        
    except FileNotFoundError:
        print(f"Error: File '{runs_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def create_sample_run() -> Dict[str, Any]:
    """
    Create a sample run in the correct format for testing.
    
    Returns:
        Sample run dictionary
    """
    return {
        "metadata": {
            "team_id": "SCIAI",
            "run_id": "SCIAI-run-example",
            "topic_id": "msmarco_v2.1_doc_04_420132660",
            "type": "automatic",
            "use_starter_kit": 0
        },
        "responses": [
            {
                "text": "This is the first sentence of the response.",
                "citations": [
                    "msmarco_v2.1_doc_04_420132660#0_1234567890",
                    "msmarco_v2.1_doc_04_420132660#1_2345678901"
                ]
            },
            {
                "text": "This is the second sentence with additional information.",
                "citations": [
                    "msmarco_v2.1_doc_04_420132660#2_3456789012"
                ]
            },
            {
                "text": "This is the third sentence providing further context.",
                "citations": []
            }
        ]
    }


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_run_format.py <runs_file.jsonl>")
        print("\nOr run without arguments to test with a sample run:")
        print("python test_run_format.py")
        sys.exit(1)
    
    runs_file = sys.argv[1]
    
    if runs_file == "sample":
        # Test with sample run
        sample_run = create_sample_run()
        print("Testing sample run format:")
        print(json.dumps(sample_run, indent=2))
        print()
        
        errors = validate_run_format(sample_run)
        if errors:
            print("Sample run validation errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ Sample run is valid!")
    else:
        # Validate runs file
        validate_runs_file(runs_file)


if __name__ == "__main__":
    main() 