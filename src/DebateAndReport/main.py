import requests
import json
import re
import os

# LangFlow API URLs
debate_pipeline_url = "http://127.0.0.1:7860/api/v1/run/9074c506-c21b-43ae-9c2b-5339cadd4164"
report_pipeline_url = "http://127.0.0.1:7860/api/v1/run/5deec78b-bd62-467c-954c-68e278b4cd6f"
report_shortener_pipeline_url = "http://127.0.0.1:7860/api/v1/run/5d1f7aa7-a685-40c4-92e4-441e6dfb8e10"
report_scorer_pipeline_url = "http://127.0.0.1:7860/api/v1/run/1e52a8ee-34a9-4669-b94b-a325d8e41533"

# Filepaths
ARTICLE_PATH = "src/DebateAndReport/input/Article1002.txt"
NOTES_PATH = "src/DebateAndReport/input/Context1002.txt"

DEBATE_LOG_PATH = "src/DebateAndReport/output/debate_log.txt"
REPORT_SCORES_PATH = "src/DebateAndReport/output/report_scores.txt"
REPORT_WITH_DEBATE_PATH = "src/DebateAndReport/output/report_with_debate.json"
REPORT_WITHOUT_DEBATE_PATH = "src/DebateAndReport/output/report_without_debate.json"

def send_request(url, payload, headers=None):
    if headers is None:
        try:
            api_key = os.environ["LANGFLOW_API_KEY"]
        except KeyError:
            raise ValueError("LANGFLOW_API_KEY environment variable not found. Please set your API key in the environment variables.")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

def generate_debate():
    article = open(ARTICLE_PATH, "r").read()
    notes = open(NOTES_PATH, "r").read()

    input_text = "Article: " + article + "\n\nNotes: " + notes

    payload = {
        "input_value": input_text,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}

    send_request(debate_pipeline_url, payload, headers)

def generate_report(use_debate):
    article = open(ARTICLE_PATH, "r").read()
    notes = open(NOTES_PATH, "r").read()
    # Ensure debate log file exists
    if not os.path.exists(DEBATE_LOG_PATH):
        open(DEBATE_LOG_PATH, "w").close()
    debate = open(DEBATE_LOG_PATH, "r").read()

    input_text = "Article: " + article + "\n\nNotes: " + notes

    if use_debate:
        input_text += """\n\nHere is the debate on the credibility of the article.\n        Insights from this debate should be your main reference point: """ + debate

    payload = {
        "input_value": input_text,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}

    response = send_request(report_pipeline_url, payload, headers)
    if response is None:
        print("Failed to get a response from the report pipeline API.")
        return
    print("API raw response:", repr(response.text))  # Debug print
    try:
        report_text = json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"]
    except Exception as e:
        print(f"Error parsing report pipeline response: {e}")
        return
    report_text = clean_json_string(report_text)
    open(REPORT_WITH_DEBATE_PATH if use_debate else REPORT_WITHOUT_DEBATE_PATH, "w").write(report_text)

def shorten_report(report_path):
    report = open(report_path, "r").read()

    input_text = report

    payload = {
        "input_value": input_text,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}

    response = send_request(report_shortener_pipeline_url, payload, headers)
    if response is None:
        print(f"Failed to get a response from the report shortener API for {report_path}.")
        return
    try:
        shortened_text = json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"]
    except Exception as e:
        print(f"Error parsing report shortener response: {e}")
        return
    shortened_text = clean_json_string(shortened_text)
    open(report_path, "w").write(shortened_text)

def score_reports(report_path_1, report_path_2):
    report_1 = open(report_path_1, "r").read()
    report_2 = open(report_path_2, "r").read()

    input_text = "Report 1\n" + report_1 + "\n\nReport 2\n" + report_2

    payload = {
        "input_value": input_text,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}

    response = send_request(report_scorer_pipeline_url, payload, headers)

def clear_reports():
    open(REPORT_WITH_DEBATE_PATH, "w").write("")
    open(REPORT_WITHOUT_DEBATE_PATH, "w").write("")

def get_word_count(responses):
        wordCount = 0
        for response in responses:
            text = response["text"]
            words = re.findall(r"\w+", text)
            wordCount += len(words)
        return wordCount

def clean_json_string(json_str):
    # Remove markdown code block indicators if present
    json_str = json_str.strip()
    if json_str.startswith('```json'):
        json_str = json_str[len('```json'):].lstrip()
    if json_str.startswith('```'):
        json_str = json_str[len('```'):].lstrip()
    if json_str.endswith('```'):
        json_str = json_str[:-3].rstrip()
    return json_str

def main():
    generate_debate()
    print("Debate Generated!")

    clear_reports()

    print("Generating report using debate...")
    generate_report(True)

    print("Generating report without debate...")
    generate_report(False)

    print("Shortening reports if needed...")

    # Handle report with debate
    try:
        report_with_debate_content = open(REPORT_WITH_DEBATE_PATH, "r").read()
        report_with_debate_content = clean_json_string(report_with_debate_content)
        if not report_with_debate_content.strip():
            print("Report with debate is empty or invalid, skipping shortening.")
        else:
            report_with_debate_data = json.loads(report_with_debate_content)
            while get_word_count(report_with_debate_data) > 250:
                shorten_report(REPORT_WITH_DEBATE_PATH)
                report_with_debate_content = open(REPORT_WITH_DEBATE_PATH, "r").read()
                report_with_debate_content = clean_json_string(report_with_debate_content)
                report_with_debate_data = json.loads(report_with_debate_content)
    except json.JSONDecodeError:
        print("Report with debate is not valid JSON, skipping shortening.")

    # Handle report without debate
    try:
        report_without_debate_content = open(REPORT_WITHOUT_DEBATE_PATH, "r").read()
        report_without_debate_content = clean_json_string(report_without_debate_content)
        if not report_without_debate_content.strip():
            print("Report without debate is empty or invalid, skipping shortening.")
        else:
            report_without_debate_data = json.loads(report_without_debate_content)
            while get_word_count(report_without_debate_data) > 250:
                shorten_report(REPORT_WITHOUT_DEBATE_PATH)
                report_without_debate_content = open(REPORT_WITHOUT_DEBATE_PATH, "r").read()
                report_without_debate_content = clean_json_string(report_without_debate_content)
                report_without_debate_data = json.loads(report_without_debate_content)
    except json.JSONDecodeError:
        print("Report without debate is not valid JSON, skipping shortening.")

    print("Scoring reports...")

    score_reports(REPORT_WITH_DEBATE_PATH, REPORT_WITHOUT_DEBATE_PATH)

    print("Finished!")


if __name__ == "__main__":
    main()
