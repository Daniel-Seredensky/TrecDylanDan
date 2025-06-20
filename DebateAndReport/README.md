# Debate-Augmented Report Generator

This project generates and evaluates credibility reports on news articles by comparing outputs from large language models with and without debate-based reasoning. The system uses several LangFlow API pipelines to simulate a debate, generate reports, shorten long outputs, and compare results.

## Project Structure

```
project/
│
├── input/
│   ├── article.txt         # News article text
│   └── notes.txt           # Analyst notes or supporting context
│
├── output/
│   ├── debate_log.txt               # Stores the debate output
│   ├── report_with_debate.txt      # Report generated using the debate
│   ├── report_without_debate.txt   # Report generated without using the debate
│   └── report_scores.txt           # Placeholder for scoring results
│
├── main.py                # Main script for running the pipeline
└── README.md              # Project documentation
```

## How It Works

1. **Debate Generation**
   A LangFlow pipeline creates a debate based on the article and notes provided.

2. **Report Generation**
   Two reports are generated. One incorporates debate results as justification. The other does not.

3. **Shortening**
   Each report is shortened if its length exceeds 250 words using a separate LangFlow shortening pipeline.

4. **Scoring**
   A final pipeline compares the two reports. This step is included in the script but does not currently write results to disk.

## Requirements

* Python 3.7 or later
* requests library

To install the required dependency:

```
pip install requests
```

## Usage

Before running the script, place your input files in the `input` directory:

* `article.txt`: The article to be evaluated
* `notes.txt`: Supplementary notes or observations generated by previous steps in pipeline

To run the script:

```
python main.py
```

The script performs the following steps in sequence:

* Generates a debate
* Produces a report that uses the debate
* Produces a second report without the debate
* Shortens both reports if necessary
* Sends both reports for comparative scoring

## API Configuration

The LangFlow API endpoints are currently configured for local use at:

```
http://127.0.0.1:7860/api/v1/run/...
```

If you are using a remote deployment or different flow IDs, you will need to update the corresponding URLs in `main.py`.

## Additional Notes

* The script expects the report outputs to be in JSON format with nested keys leading to a text field. This structure is based on LangFlow's default response.
* The file `report_scores.txt` is reserved but not used. Scoring results are currently printed to the console or ignored depending on the implementation.