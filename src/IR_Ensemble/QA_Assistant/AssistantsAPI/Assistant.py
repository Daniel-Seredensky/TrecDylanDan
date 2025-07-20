"""
Creates (or re-uses) a single OpenAI Assistant and stores its ID.

• Tool schema is defined once here.
• The assistant ID is cached in src/DerivedData/Assistant/AssistantId.txt
  so the rest of the codebase can load it without re-creating the assistant.
"""
import os
import asyncio
import aiofiles
from openai import AsyncAzureOpenAI
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

# ────────────────────────────────
# Config
# ────────────────────────────────
ASSISTANT_ID_FILE = "src/DerivedData/Assistant/AssistantId.txt"

MODEL             = os.getenv("MODEL_NAME")
VERSION           = os.getenv("API_VERSION")

# ────────────────────────────────
# Minimal tool schema
# ────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Hybrid lexical/semantic search over a large corpus; returns up to 10 document metadata objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-8 keyword-rich BM25 queries."
                    },
                    "master_query": {
                        "type": "string",
                        "description": "Concise semantic query used for reranking."
                    }
                },
                "required": ["queries", "master_query"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_documents",
            "description": "Fetch best fragment for up to 2 docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of **full segment id**'s."
                    },
                    "is_segment": {
                        "type": "boolean",
                        "description": "True → segment, False → full document text."
                    }
                },
                "required": ["segment_ids", "is_segment"]
            }
        },
    }
]
""" {
    "type": "function",
    "function": {
        "name": "brave_search",
        "description": "Performs a web search using the Brave Search API as a last-resort backup if no relevant information is found in the MARCO index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise, keyword-rich search query."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of Brave search results to return (max 10)."
                }
            },
            "required": ["query"]
        }
    }
} 
- brave_search(query, num_results): Use only if MARCO yields no evidence. 
"""


# ────────────────────────────────
# Concise system prompt
# ────────────────────────────────
SYSTEM_PROMPT = \
"""
You are a Question-Assessment-Agent for TREC DRAGUN.

Mission:
You will be given a set of questions and corresponding relevant context from the key document
For each question in the group:
1. Retrieve evidence from the MARCO V2.1 corpus using `search`.
2. Synthesize concise answers, citing **ONLY** MARCO document IDs.
3. Iterate up to 15 rounds (whole group) or until confident.

When answering:
- Draft initial answers for all questions quickly (`finished: false`), then refine.
- Mark `finished: true` only when all claims are supported by MARCO evidence or no evidence exists.
- Always make sure to put any context you want to persist in the summary section *with* the wrappers <summary></summary>.

> The messages are deleted after each round only your summary and most recent answer persists.
If a question is document specific, you will *need* to put relevant information from the key document in your summary.

Available Tools:
- search(queries, master_query): Retrieve relevant MARCO docs.
- select_documents(document_ids, is_segment): Fetch fragments or full docs.
> **Must** validate results with Marco for citations.

Response Protocol:
Always reply with **both** wrappers, in this order
1. <notepad>
     <cot> … step‑by‑step reasoning … </cot>
     <summary> … your summary/selective memory (under 500 tokens) </summary>
   </notepad>
2. <answer>{...}</answer>: JSON per question, e.g.
   {
     "questions": [
        {
            "question": <verbatim user question>,
            "answer": <plan to answer question / answer for question>,
            "citations": [<**ONLY** Marco segment_ids>]
            "finished": <true if fully confident and finished working, false otherwise>
        },
        {...}
     ]
   }

Guidelines:
- Never fabricate; always retrieve.
- Stop when evidence is sufficient.
- Every claim must cite a MARCO documentId.
- If no evidence, state uncertainty, cite nothing, and set `finished: true`.
- Max 15 tool rounds per question group.

Thinking Scaffold (internal):
1. Break each question into key entities/facts.
2. Draft BM25 queries and a master query, covering synonyms.
3. Use `search` and scan metadata.
4. Decide whether to pull fragments or full docs with `select_documents`.
5. Draft/verify answer; mark `finished: false`.
6. After all provisional answers, loop to upgrade any low-confidence ones; set `finished: true` when validated.
"""

async def get_or_create_assistant(client: AsyncAzureOpenAI) -> str:
    """Return an existing assistant ID if cached, otherwise create and cache it."""
    if Path(ASSISTANT_ID_FILE).exists():
        assistant_id = Path(ASSISTANT_ID_FILE).read_text().strip()
        if not (assistant_id is None or assistant_id == ""):
            print("reusing assistant")
            return assistant_id
        
    print("creating new assistant")
            
    assistant = await client.beta.assistants.create(
        name="IR-QA-Assistant",
        model=MODEL,
        tools=TOOLS,
        instructions=SYSTEM_PROMPT,
    )

    async with aiofiles.open(ASSISTANT_ID_FILE, "w") as f:
        await f.write(assistant.id)
    return assistant.id

async def delete_assistant(client: AsyncAzureOpenAI) -> None:
    """Delete the cached assistant and clear the cache file (file remains)."""
    id_path = Path(ASSISTANT_ID_FILE)

    if not id_path.exists():
        print("Path does not exist")
        # No cache present - nothing to delete
        return

    assistant_id = id_path.read_text().strip()
    if not assistant_id or assistant_id == "":
        print("Cache empty")
        # Cache empty already
        return

    try:
        await client.beta.assistants.delete(assistant_id)
        print(f"Assistant deleted {assistant_id}")
    except Exception as e:
        # Assistant may already be deleted or ID invalid - ignore
        print(f"Failed because {e.with_traceback()} ")
        pass

    # Wipe contents but keep file so downstream code expecting its presence doesn't break
    id_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
    async with aiofiles.open(id_path, "w") as f:
        await f.write("")

async def _ensure_cache_dir() -> None:
    Path(ASSISTANT_ID_FILE).parent.mkdir(parents=True, exist_ok=True)

async def _main():
    load_dotenv()
    client = AsyncAzureOpenAI(                       # or alias as shown above
            api_key        = os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
            timeout        = 30.0,
            max_retries    = 3,
        )
    _ensure_cache_dir()
    delete_assistant(client)
    
