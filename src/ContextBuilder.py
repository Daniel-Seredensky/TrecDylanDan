import json
import os
import asyncio 
import aiofiles
from asyncinit import asyncinit

from openai import AsyncOpenAI
from dotenv import load_dotenv; load_dotenv()

from QA_Assistant.QuestionEval import assess_question
from QA_Assistant.Assistant import get_or_create_assistant

@asyncinit
class Proctor:
    async def __init__(self, docId: str) -> None:
        """
        Takes in a document and does research and creates the context for the debate
        """
        self.context_path = os.getenv("CONTEXT_PATH")
        self.document = docId
        self.client = AsyncOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
            )
        # not implemented yet
        self.template_path = await get_q_template(self.document)
        try :
            self.assistantId = await get_or_create_assistant(self.client)
        except: 
            print("Error creating assistant")
            self.client.close()
            exit(1)

    async def create_context (self) -> str:
        with aiofiles.open(self.template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
            groups = template["groups"]
            tasks = []
            for section in groups:
                questions = section["questions"]
                tasks.append(assess_question(questions, self.document, self.client, self.assistantId))
            with aiofiles.open(self.context_path, 'w', encoding='utf-8') as f:
                async for r in await asyncio.gather(*tasks, return_exceptions=True):
                    json.load(r)
                    await f.write(f"Questions: \n{r["question"]}\n\n Answer: \n{r["answer"]}\n\n Citations: \n{r["citations"]}")
        return self.context_path
    
    
