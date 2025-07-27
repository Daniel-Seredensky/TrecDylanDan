from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv
import traceback

import os
import asyncio, uvloop

from src.IR_Ensemble.QA_Assistant.bucket_monitor import BucketMonitor
from src.IR_Ensemble.QA_Assistant.Searcher import cohere_client
from src.IR_Ensemble.QA_Assistant.daemon_wrapper import JVMDaemon 
from src.IR_Ensemble.context_builder import ContextProctor
from src.ReportGenerator.report_generator import ReportGenerator
from src.ReportEvaluator.report_evaluator import ReportEvaluator, EvalStatus
from topics import topic1

load_dotenv()
MAX_ROUNDS = 3

async def get_context(client: AsyncAzureOpenAI, questions: list[dict[str,str]]) -> str:
    """
    Get search results for a set of questions.
    """
    proc = ContextProctor(client, questions)
    try: 
        await proc.create_context()
        # clear mem
        del proc
        with open(os.getenv("CONTEXT_PATH"), "r") as f: return f.read()
    except Exception as e:
        print(e)
        traceback.print_exc()
        del proc
        return "Error generating context."

async def _main(oai_client: AsyncOpenAI,aoai_client: AsyncAzureOpenAI,topic: str) -> str:
    # clear old context if it exists else ensure it exists
    with open(os.getenv("CONTEXT_PATH"), "w") as f: f.write("")

    # init agents
    gen: ReportGenerator = ReportGenerator(client = oai_client, topic = topic)
    eval: ReportEvaluator = ReportEvaluator(client = oai_client, topic = topic)
    rounds = 0
    note,context = [None]*2

    # run loop
    while rounds < MAX_ROUNDS:
        report,note = await gen.generate_report(context, note)
        note,questions = await eval.evaluate(report = report, generator_comment = note, ir_context = context)
        if eval.status == EvalStatus.PASS:
            break
        context = await get_context(client = aoai_client, questions = questions)
        rounds += 1
    return report 

async def main():
    #segment_id = "<smth>"
    #topic = JVMDaemon.select_documents([segment_id],is_segment = False)
    topic = topic1
    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout = 60,
        max_retries = 5,
    )
    aoai_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="preview",
        timeout=120,
        max_retries=5,
    )
    bm = BucketMonitor()
    try: 
        await bm.start()
        return await _main(oai_client=oai_client,
                           aoai_client=aoai_client,
                           topic=topic)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        await oai_client.close()
        await aoai_client.close()
        await cohere_client.aclose()
        await JVMDaemon.stop()
        await bm.stop()

if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    report = asyncio.run(main())
    print("\n=== FINAL REPORT ===\n")
    print(report)