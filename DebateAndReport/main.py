import requests
import json
import report_generator as ReportGenerator
import report_scorer as ReportScorer

url = "http://127.0.0.1:7862/api/v1/run/ff071340-7374-4b9b-b222-bb157ff5be7b"  # The complete API endpoint URL for this flow

def write_report(includeDebate, path):
    if includeDebate:
        file = open("debate_log.txt", "r")
        debate_log_text = "Here is the debate on the credibility of the article. Insights from this debate should be your main reference point: " + file.read()
        file = open(path, "w")
        report_generator = ReportGenerator.ReportGenerator()
        file.write(report_generator.generate_report(
            article=article_text,
            notes=notes_text,
            additional=debate_log_text,
        )
    )
    else:
        file = open(path, "w")
        report_generator = ReportGenerator.ReportGenerator()
        file.write(report_generator.generate_report(
            article=article_text,
            notes=notes_text,
            additional="",
        )
    )
    return

# Process article and notes inout

ARTICLE_PATH = "input/article.txt"
NOTES_PATH = "input/notes.txt"

input_value = ""
article_text = ""
notes_text = ""

with open(ARTICLE_PATH, "r") as f:
    article_text = f.read()
    input_value += "ARTICLE:\n\n" + article_text

with open(NOTES_PATH, "r") as f:
    notes_text = f.read()
    input_value += "\n\nNOTES:\n\n" + notes_text



# Request payload configuration
payload = {
    "input_value": input_value,  # The input value to be processed by the flow
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

    # Save response as .json
    data = json.loads(response.text)
    #file = open("response.json", "w")
    #file.write(response.text)

    # Final output text
    # output = (data["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"])
    # file = open("output.txt", "w")
    # file.write(output)
    
    write_report(True, "report_with_debate.json")
    write_report(False, "report_without_debate.json")

    report_scorer = ReportScorer.ReportScorer()
    report_score = open("report_scores.txt", "w")
    report1 = open("report_with_debate.json", "r").read()
    report2 = open("report_without_debate.json", "r").read()
    report_score.write(report_scorer.scoreReports(report1, report2, article_text))

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")