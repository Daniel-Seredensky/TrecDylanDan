import requests
import json

debate_pipeline_url = "http://127.0.0.1:7860/api/v1/run/91b0b37f-1f48-4933-ad65-94950bc22ad5"
report_pipeline_url = ""
report_scorer_pipeline_url = ""

def send_request(url, payload, headers):
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

def generate_debate():
    ARTICLE_PATH = "input/article.txt"
    NOTES_PATH = "input/notes.txt"

    with open(ARTICLE_PATH, "r") as f:
        article = f.read()
    with open(NOTES_PATH, "r") as f:
        notes = f.read()

    input_text = "Article: " + article + "\n\nNotes: " + notes

    payload = {
        "input_value": input_text,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}

    send_request(debate_pipeline_url, payload, headers)

def main():
    generate_debate()

if __name__ == "__main__":
    main()
