from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import traceback

import os
import asyncio, uvloop, json

from src.IR_Ensemble.QA_Assistant.bucket_monitor import BucketMonitor
from src.IR_Ensemble.QA_Assistant.Searcher import cohere_client
from src.IR_Ensemble.QA_Assistant.daemon_wrapper import JVMDaemon 
from src.IR_Ensemble.context_builder import ContextProctor
from src.ReportGenerator.report_generator import ReportGenerator
from src.ReportEvaluator.report_evaluator import ReportEvaluator, EvalStatus

load_dotenv()
MAX_ROUNDS = 3

async def get_context(client: AsyncAzureOpenAI, questions: list[dict[str,str]], num: int) -> str:
    """
    Get search results for a set of questions.
    """
    proc = ContextProctor(client, questions, num)
    try: 
        await proc.create_context()
        # clear mem
        with open(proc.context_path, "r") as f: return f.read()
        del proc
    except Exception as e:
        print(e)
        traceback.print_exc()
        del proc
        raise RuntimeError("Failed to get context")

async def _main(gen_client: AsyncAzureOpenAI,
                ir_client: AsyncAzureOpenAI,
                topic: dict[str,str],
                  num: int) -> str:
    # extract marco id from topic and ensure it exists
    id = topic.get("docid", None)
    if not id: raise ValueError("Topic must contain a 'docid' key with the segment ID.")
    # convert topic to string
    topic = json.dumps(topic, ensure_ascii=True)

    # init agents
    gen: ReportGenerator = ReportGenerator(client = gen_client,
                                            topic = topic,
                                              num = num)
    eval_: ReportEvaluator = ReportEvaluator(client = gen_client,
                                             topic = topic,
                                             num = num)
    rounds = 0
    note,context = [None]*2
    questions = []
    evaluation = "No evaluation yet"

    # run loop
    while rounds < MAX_ROUNDS:
        if rounds == 0:
            pass
        else:
            context = await get_context(client = ir_client, questions = questions, num = num)
        report,note = await gen.generate_report(context, note, evaluation)
        note,questions,evaluation = await eval_.evaluate(report = report, generator_comment = note, ir_context = context)
        if eval_.status == EvalStatus.PASS:
            break
        rounds += 1
    return {"id":id,"report":eval_.best['report'], "score": eval_.best['score']}

async def main(topics: list[dict[str,str]]) -> None:
    API_VER = "2025-04-01-preview"
    gen_client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=API_VER,
        timeout=120,
        max_retries=5,
    )

    ir_client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["IR_AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["IR_AZURE_OPENAI_KEY"],
        api_version=API_VER,
        timeout=120,
        max_retries=5,
    )

    bm = BucketMonitor()
    try: 
        await bm.start()
        tasks = [_main(gen_client=gen_client,
                           ir_client=ir_client,
                           topic=topic,
                           num=num)
                    for num,topic in enumerate(topics)]
        results = await asyncio.gather(*tasks)
        # write results to file
        with open("RES.txt","a", encoding="utf-8") as f:
            f.write(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        await gen_client.close()
        await ir_client.close()
        await cohere_client.aclose()
        await JVMDaemon.stop()
        await bm.stop()

if __name__ == "__main__":
    count = 0
    topics = []
    with open("trec-2025-dragun-topics.jsonl", "r") as f:
        for line in f:
            # next 2,4
            if not (count >= 0 and count < 10): 
                count += 1
                continue
            else:
                topics.append(json.loads(line))
            count += 1
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main(topics=topics))
    