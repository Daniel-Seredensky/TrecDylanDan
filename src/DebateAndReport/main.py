import requests
import json
import re

# LangFlow API URLs
debate_pipeline_url = "http://127.0.0.1:7860/api/v1/run/ff071340-7374-4b9b-b222-bb157ff5be7b"
report_pipeline_url = "http://127.0.0.1:7860/api/v1/run/67ea3993-b394-4115-a3a4-b51a329a7e28"
report_shortener_pipeline_url = "http://127.0.0.1:7860/api/v1/run/05043cdc-403e-4356-bf96-192e8b487f5b"
report_scorer_pipeline_url = "http://127.0.0.1:7860/api/v1/run/3315485d-c0de-46fb-ac03-291563d1e6a1"

# Filepaths
ARTICLE_PATH = "DebateAndReport/input/article.txt"
NOTES_PATH = "DebateAndReport/input/notes.txt"

DEBATE_LOG_PATH = "DebateAndReport/output/debate_log.txt"
REPORT_SCORES_PATH = "DebateAndReport/output/report_scores.txt"
REPORT_WITH_DEBATE_PATH = "DebateAndReport/output/report_with_debate.txt"
REPORT_WITHOUT_DEBATE_PATH = "DebateAndReport/output/report_without_debate.txt"

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
    try:
        report_text = json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"]
    except Exception as e:
        print(f"Error parsing report pipeline response: {e}")
        return
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
            words = re.findall("\w+", text)
            wordCount += len(words)
        return wordCount

def main():
    generate_debate()
    print("Debate Generated!")

    clear_reports()

    print("Generating report using debate...")
    generate_report(True)

    print("Generating report without debate...")
    generate_report(False)

    print("Shortening reports if needed...")

    while get_word_count(json.loads(open(REPORT_WITH_DEBATE_PATH, "r").read())) > 250:
        shorten_report(REPORT_WITH_DEBATE_PATH)

    while get_word_count(json.loads(open(REPORT_WITHOUT_DEBATE_PATH, "r").read())) > 250:
        shorten_report(REPORT_WITHOUT_DEBATE_PATH)

    print("Scoring reports...")

    score_reports(REPORT_WITH_DEBATE_PATH, REPORT_WITHOUT_DEBATE_PATH)

    print("Finished!")


if __name__ == "__main__":
    main()
