import requests
import json
from pathlib import Path
from typing import Iterable, Union
import re

url = "http://127.0.0.1:7860/api/v1/run/b0187b26-3953-4842-8445-c342a447ea79"  # The complete API endpoint URL for this flow
    
def combine_group_json_texts(json_texts: Iterable[str],indent: int | None = 2,) -> str:
    merged = {"groups": []}

    for jtxt in json_texts:
        data = json.loads(jtxt)
        merged["groups"].extend(data.get("groups", []))

    return json.dumps(merged, ensure_ascii=False, indent=indent)

def _extract_text_field(obj: dict) -> str:
    for key in ("segment", "segment_text", "contents", "text", "Clean-Text", "body"):
        if key in obj and obj[key]:
            return obj[key]
    return ""

def get_document_text(doc_or_segment_id: str, jsonl_path: str) -> str:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    base_id = doc_or_segment_id.split('#')[0]
    segments = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seg_id = obj.get("segment_id") or obj.get("docid") or obj.get("id")
            if not seg_id:
                continue
            if seg_id.split('#')[0] == base_id:
                m = re.match(r".*#(\d+)", seg_id)
                seg_idx = int(m.group(1)) if m else 0
                text = _extract_text_field(obj)
                segments.append((seg_idx, text))
    segments.sort()
    doc_text = "\n".join(text for _, text in segments if text)
    return doc_text

def create_question_template(document_id):
    # Get text from document id
    document_text = get_document_text(document_id, "DebateAndReport/FixedSampleData.jsonl")

    # Request payload configuration
    payload = {
        "input_value": document_text,  # The input value to be processed by the flow
        "output_type": "text",  # Specifies the expected output format
        "input_type": "text"  # Specifies the input format
    }

    # Request headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        return combine_group_json_texts(
            [json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"],
             json.loads(response.text)["outputs"][0]["outputs"][1]["results"]["text"]["data"]["text"],
             json.loads(response.text)["outputs"][0]["outputs"][2]["results"]["text"]["data"]["text"]
            ])

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

if __name__ == "__main__":
    # Example usage for testing
    sample_doc_id = "msmarco_v2.1_doc_17_795452723#14_866362073"
    try:
        # Directly test document extraction and print result
        doc_text = get_document_text(sample_doc_id, "DebateAndReport/FixedSampleData.jsonl")
        print("[MAIN DEBUG] Full extracted document text (first 500 chars):\n", doc_text[:500], "\n---\n")
        # Now run the question template generator
        questions_json = create_question_template(sample_doc_id)
        print("\nGenerated Questions JSON:\n", questions_json)
        # Write the question template output to a file
        output_path = Path(__file__).parent / "question_template_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(questions_json if questions_json is not None else "")
        print(f"[MAIN DEBUG] Wrote question template output to {output_path}")
    except Exception as e:
        print(f"Error: {e}")