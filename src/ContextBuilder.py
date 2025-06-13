import json
import os
import asyncio 
from asyncinit import asyncinit

from openai import AsyncOpenAI
from dotenv import load_dotenv; load_dotenv()

from src.InfoRetrieval.QuestionEval import assess_question


@asyncinit
class Proctor:
    MAP = {0:"Political.json", 
           1:"Scientific.json", 
           2:"Health.json", 
           3:"Business.json", 
           4:"Technology.json", 
           5:"Investigative.json", 
           6:"Opinion.json", 
           7:"Data-Journalism.json"}
    
    async def __init__(self, docId: str) -> None:
        self.document = docId
        client = await AsyncOpenAI(api_key=os.getenv("API_KEY"))
        # bin document and retrieve corresponding template
        # template generation will likely be async
        await bin_document(client, docId)
        # something like await get_template(client, docId)
        with open(os.getenv("BIN_AGENT_RESULTS_PATH"), 'r', encoding='utf-8') as f:
            claim_data = json.load(f)
            self.bin = claim_data["bin_number"]
            self.templatePath = f"Templates/{self.MAP[self.bin]}"
            
    async def create_context (self, contextPath: str) -> str:
        with open(self.templatePath, 'r', encoding='utf-8') as f:
            template = json.load(f)
            tasks = []
            for _, question in template.items():
                tasks.append(assess_question(question, self.document))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            with open(contextPath, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(r.question + "\n" + r.answer + "\n\n")
            await reset_index()
        return contextPath
    
    # sync wrapper for starting async context generation event loop
    # for testing purposes, the async event loop will be initialized in the main file
    def _createContext (self,path) -> str:
        return asyncio.run(self.create_context(path))
    
async def sub_main():
    docId = "clueweb22-en0024-53-03398"
    p = await Proctor(docId)
    return await p.create_context("context.txt")

def main():
    print(asyncio.run(sub_main()))

if __name__ == "__main__":
    main()
