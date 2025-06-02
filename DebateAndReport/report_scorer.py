import os
from openai import OpenAI

class ReportScorer:
    def scoreReports(self, report1, report2, article):
        response = OpenAI().responses.create(
            model= "o4-mini",
            input= "Report 1: " + report1 + "\n\n Report 2: " + report2 + "\n\n Article: " + article,
            instructions = """
    You are a professional fact-checker and media literacy expert.
    Your ultimate task is to assess two reports that are supposed to be well-attributed and providing background and context to help readers assess the trustworthiness of a given news article.
    Based on the source article and these two reports, you must score both of these reports from 1-100 and provide explicit reasoning behind your scoring.
""",
        )
        return response.output[1].content[0].text
