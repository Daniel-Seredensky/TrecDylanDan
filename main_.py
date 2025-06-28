from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import traceback

import os
import asyncio
import json

from src.QA_Assistant.DocSelect import select_documents
from src.QA_Assistant.Assistant import get_or_create_assistant, delete_assistant
from src.QA_Assistant.QuestionEval import assess_question
from pathlib import Path
from src.ContextBuilder import main as __main
from src.QA_Assistant.rate_limits import (
    openai_req_limiter,
    _global_tok_limiter,
    cohere_rerank_limiter
)
from src.QA_Assistant.bucket_monitor import BucketMonitor

async def _main():
    load_dotenv(dotenv_path=Path(".env"), override= True)
    api_key        = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version    = os.getenv("AZURE_API_VERSION")
    print(api_key,azure_endpoint,api_version)
    try:
        bucket_path = os.getenv("BUCKET_USAGE_PATH")
        bm = BucketMonitor(
            openai_req_bucket= openai_req_limiter,
            openai_tok_bucket= _global_tok_limiter,
            cohere_req_bucket= cohere_rerank_limiter,
            csv_path= f"{bucket_path}/bucket_usage.csv"
        )
        await bm.start()
        questions = "\n".join(json.dumps(q) for q in [
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
        ])
        client = AsyncAzureOpenAI(                       # or alias as shown above
            api_key        = os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
            timeout        = 125.0,
            max_retries    = 3,
        )
        assistant = await get_or_create_assistant(client)
        s = await assess_question(question= questions, client= client, assistant_id= assistant)
    except Exception as e:
        print (traceback.print_exc())
    finally:
        await client.close()
        await bm.stop()

def main():
    asyncio.run(_main())

async def _delete_assistant():
    client = AsyncAzureOpenAI(                       # or alias as shown above
            api_key        = os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
            timeout        = 200,
            max_retries    = 3,
        )
    try:
        await delete_assistant(client= client)
    finally:
        await client.close()

if __name__ == "__main__":
    main()
    
        


        
