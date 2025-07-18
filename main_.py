from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import traceback

import os
import asyncio, aiofiles, uvloop
import json

from src.QA_Assistant.bucket_monitor import BucketMonitor
from src.QA_Assistant.question_eval import assess_questions
from src.QA_Assistant.Searcher import cohere_client
from src.ContextBuilder import test
from src.QA_Assistant.daemon_wrapper import JVMDaemon 

async def _main():
    load_dotenv(override=True)
    questions = "\n".join([json.dumps(q) for q in [
            {
            "question": "Summarize the historical context and significance of the Nord Stream 2 pipeline project.",
            "doc_context": "The text notes the pipeline is \"controversial,\" aims to bypass Ukraine, and that U.S.–German disputes date back to the Siberian pipeline crisis of the 1980s. It references unchanged sanctions since 2014 but gives no timeline of planning, financing consortium, construction milestones, or previous legal challenges—information needed for a full historical summary."
            },
            {
            "question": "Describe the historical context of the Nord Stream 2 project and why it is considered controversial.",
            "doc_context": "The article brands Nord Stream 2 controversial due to security fears and vague safeguards, citing Russia’s potential energy coercion and Ukraine’s vulnerabilities. It does not cover earlier controversies such as EU antitrust debates, environmental objections, or the pipeline’s origin alongside Nord Stream 1, leaving historical context incomplete."
            },
            {
            "question": "Assess whether the article provides sufficient context about the geopolitical implications of Nord Stream 2 for U.S.-Russia and EU-Russia relations.",
            "doc_context": "The piece discusses U.S.–Germany tensions and references sanctions but gives limited detail on broader U.S.–Russia or EU‑Russia energy dynamics, NATO considerations, or prior gas disputes. Additional sources are required for a comprehensive assessment."
            }
        ]])
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="preview",
        timeout=60.5,
        max_retries=3,
    )
    try:
        await assess_questions(questions,client=client)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        await client.close()

async def newTest():
    load_dotenv()
    try:
        async with aiofiles.open(os.getenv("CONTEXT_PATH"), "w") as f:
            await f.write("")
        bm = BucketMonitor()
        daemon = JVMDaemon()
        await daemon._start()
        await bm.start()
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="preview",
            timeout=60.5,
            max_retries=3,
        )
        await test(client=client)
    finally: 
        await JVMDaemon.stop()
        await bm.stop()
        await client.close()
        await cohere_client.aclose()

if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(newTest())
