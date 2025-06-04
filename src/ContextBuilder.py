import json
import os
import asyncio 

from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()

from src.InfoRetrieval.BinAgent import bin_document
from src.InfoRetrieval.QuestionEval import assess_question

class Proctor:
    MAP = {0:"Political.json", 
           1:"Scientific.json", 
           2:"Health.json", 
           3:"Business.json", 
           4:"Technology.json", 
           5:"Investigative.json", 
           6:"Opinion.json", 
           7:"Data-Journalism.json"}

    def __init__(self, docId: str) -> None:
        self.document = docId
        client = OpenAI(api_key=os.getenv("API_KEY"))
        # bin document and retrieve corresponding template
        bin_document(client, docId)
        with open(os.getenv("BIN_AGENT_RESULTS_PATH"), 'r', encoding='utf-8') as f:
            claim_data = json.load(f)
            self.bin = claim_data["bin_number"]
            self.templatePath = f"Templates/{self.MAP[self.bin]}"
            
    async def createContext (self, contextPath: str) -> str:
        with open(self.templatePath, 'r', encoding='utf-8') as f:
            template = json.load(f)
            tasks = []
            for _, question in template.items():
                tasks.append(assess_question(question, self.document))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            with open(contextPath, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(r.question + "\n" + r.answer + "\n\n")
        return contextPath
    

