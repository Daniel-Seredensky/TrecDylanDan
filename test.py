
from enum import Enum
class LoopStage(Enum):
    """
    Enum for the different stages of the loop
    each stage corresponds to a different set of responses parameters and a absolute max reservation size
    there is also a unique identifier for each stage at the end of its list so that LoopStages will never equal each other
    eg 
    python: 
    >>> LoopStage.PLAN_CALL.value
    [{"max_output_tokens":3_500,"model": "gpt-4.1","previous_response_id": None}, 50_000,1]
    """
    # Previous response id should be None for PLAN call always
    PLAN_CALL = [{"max_output_tokens":3_500,
                  "model": "gpt-4.1",
                  "previous_response_id": None,
                  "temperature":0.4,
                  "top_p":0.95}, 50_000, 1]
    TOOL_CALL = [{"max_output_tokens":2_000,
                  "model": "gpt-4.1-mini",
                  "previous_response_id": None,
                  "temperature":0.2,
                  "top_p":0.9}, 100_000,2]
    UPDATE_CALL = [{"max_output_tokens":5_000,
                    "model": "gpt-4.1-mini",
                    "previous_response_id": None,
                    "temperature":0.25,
                    "top_p":0.9}, 100_000,3]
    FINAL_CALL = [{"max_output_tokens":1_000,
                   "model": "gpt-4.1-mini",
                   "previous_response_id": None,
                   "temperature":0.4,
                   "top_p":0.95}, 100_000,4]
    
import json

test = {"foo":"bar","test":1}
q = json.loads(json.dumps(test))
print(q["foo"])