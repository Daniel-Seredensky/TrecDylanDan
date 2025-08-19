import traceback
from typing import List,Dict,Any
from openai import AsyncAzureOpenAI
from src.IR_Ensemble.QA_Assistant.base import BaseAgent, QAStatus
from src.IR_Ensemble.QA_Assistant.rate_limits import LoopStage

class QuestionEvalAgent(BaseAgent):
    """
    Evaluate questions for relevance to the document set.
    """
    async def run(self) -> None:
        # loop until finished
        rounds = 0
        while self.status != QAStatus.FINISHED:
            segments = await self.get_info(first_round= (rounds == 0))
            await self.update_answer(segments)
            rounds += 1
            if rounds >= self.MAX_TOOL_ROUNDS:
                await self.force_final_prompt()
                break
            await self.reset_logical_thread()
        return {"summary": self.summary,  "status": self.status.name, "answer": self.full_answer}

async def assess_questions(
    questions: str,
    client: AsyncAzureOpenAI,
    num: int
) -> List[Dict[str, Any]]:
    """
    Evaluate questions for relevance to the document set.
    """
    agent = await QuestionEvalAgent(questions, client, num=num)
    try :
        return await agent.run()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        return {"summary": agent.summary, "status": agent.status.name, "answer": agent.full_answer}
    
    



