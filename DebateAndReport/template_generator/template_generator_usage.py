import requests
import json
from pathlib import Path
from typing import Iterable, Union
from src.InfoRetrieval import Utils

url = "http://127.0.0.1:7860/api/v1/run/d291fdb7-25cf-463d-a43e-a1a08e2ace49"  # The complete API endpoint URL for this flow
    
def combine_group_json_texts(json_texts: Iterable[str],indent: int | None = 2,) -> str:
    merged = {"groups": []}

    for jtxt in json_texts:
        data = json.loads(jtxt)
        merged["groups"].extend(data.get("groups", []))

    return json.dumps(merged, ensure_ascii=False, indent=indent)


def create_question_template(document_id):
    # Get text from document id
    document_text = Utils.get_document_text(document_id, "DebateAndReport/FixedSampleData.jsonl")

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