SEARCH_CONTRACT = \
"""
Given the following plan and set of questions, return a json of bm25 optimized keyword 
queries (MARCO search) and a master query (used for semantic rerank). You may have up 
to 4 [queries,master_query] pairs in your "searches" array. 

> You *MUST* answer with the following format:
``` 
<cot> Brief cot summary </cot>
<answer>
    {
        "searches":[
            {
                "queries": [
                    <query1>,
                    <query2>,
                    ...
                ],
                "master_query": "master_query"
            },
            ...
        ] 
    }
<answer>
```
"""

SELECT_CONTRACT = \
"""
Given previous context and following search result metadata 
select up to 6 segment_ids for further exploration

> You *MUST* answer with the following format
``` 
<cot> Brief cot summary </cot>
<answer>
    {
        "selections":[
            <segment_id1>,
            <segment_id2>,
            ...
        ]
    }
</answer>
```
"""

PLAN_CONTRACT = \
"""
You are an orchestrator for different RAG agents. Each RAG agent has 5 rounds with 
the following format: (search -> metadata -> select -> update answer). This RAG agent 
must work to answer the following questions. Your job is to create an efficient plan of retrieval.

> You *MUST* answer with the following format
``` 
<cot> Bried cot summary </cot>
<answer> Your constructed plan for the RAG agent</answer>
```
"""

UPDATE_CONTRACT = \
"""
Given the previous context and the search results given below update your answer status

> You *MUST* answer with the following format
make sure to append your round summary to the end of the rounds array 
making sure not to overwrite any previous round summary
``` 
<cot> Brief cot summary</cot> 
<answer>
   {
     "questions": [
        {
            "question": <verbatim user question>,
            "answer": <in progress answer/ finished answer>,
            "citations": [<**ONLY** Marco segment_ids>]
            "finished": <true if fully confident and finished working, false otherwise>
        },
        ...
     ],
   "rounds": [
      {
         "summary": <summary of round, include successes and shortcomings, info that should persist,etc>
      },
      ...
   ]
}
</answer>
```
"""

FINAL_CONTRACT = \
"""
You have exceeded the number of rounds available give a brief description of what 
you attempted, what worked, what didn't, and any additional information that would be required

> You *MUST* answer with the following format
```
<cot> Bried cot summary </cot>
<answer> Your summary </answer>
"""