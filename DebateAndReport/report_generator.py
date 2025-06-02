import json
import re
from typing import List

from pydantic import BaseModel
from openai import OpenAI

# --------------------------------------------------------------------------- #
# MODELS
# --------------------------------------------------------------------------- #
class Sentence(BaseModel):
    rationale: str
    sentence_text: str
    citations: List[str]

class Report(BaseModel):
    sentences: List[Sentence]

# --------------------------------------------------------------------------- #
# PROMPTS
# --------------------------------------------------------------------------- #
OUTPUT_FORMAT = """
[
    {
        "text": "This is the first sentence.",
        "citations": ["msmarco_v2.1_doc_xx_xxxxxx1#x_xxxxxx3"]
    },
    {
        "text": "This is the second sentence.",
        "citations": []
    }
]
"""

SYSTEM_PROMPT = f"""
(unchanged – keep your original long system prompt text here)
"""

# --------------------------------------------------------------------------- #
# UTILITIES
# --------------------------------------------------------------------------- #
def count_report_words(report_json: str) -> int:
    """Return total words in all 'text' fields of the JSON report."""
    try:
        items = json.loads(report_json)
        return sum(len(re.findall(r"\\b\\w+\\b", item["text"])) for item in items)
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: count words in the raw string
        return len(re.findall(r"\\b\\w+\\b", report_json))

# --------------------------------------------------------------------------- #
# REPORT GENERATOR
# --------------------------------------------------------------------------- #
class ReportGenerator:
    def __init__(self, max_attempts: int = 5):
        self.client = OpenAI()
        self.max_attempts = max_attempts

    def _call_model(self, user_input: str) -> str:
        """Send a prompt to the model and return the raw text of the report."""
        response = self.client.responses.create(
            model="o4-mini",
            input=user_input,
            instructions=SYSTEM_PROMPT,
        )
        # The exact navigation may differ—adjust to your SDK’s response schema
        return response.output[1].content[0].text.strip()

    def generate_report(self, *, article: str, notes: str, additional: str = "") -> str:
        base_user_input = f"""
Here is the news article to evaluate:
{article}

Here is an outline of critical notes to address in the report:
{notes}

{additional}

Generate a report that addresses as many of the important questions as possible using only the information available in the retrieved segments. Each sentence should be factual, well-grounded, and include appropriate citations.
"""
        attempt = 0
        user_input = base_user_input

        while attempt < self.max_attempts:
            attempt += 1
            report_json = self._call_model(user_input)
            word_total = count_report_words(report_json)

            if word_total <= 250:
                return report_json  # ✅ Success

            # Add an explicit reminder for the next attempt
            user_input = (
                base_user_input
                + f"\n\nYour last output contained {word_total} words—"
                  "please regenerate a version STRICTLY under 250 words total."
            )

        raise RuntimeError(
            f"Unable to obtain a report within 250 words after {self.max_attempts} attempts."
        )
