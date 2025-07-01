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
You are an orchestrator for different RAG agents. The RAG agent has 5 rounds to answer the given questions.
Each round consists of 3 steps (search -> select -> update):
1. Generate Bm25 optimized keyword queries and a master query for semantic rerank (search)
2. Select relevant documents from metadata result (select)
3. Update answer given the selected documents (update)
Your job is to create an efficient plan of retrieval.

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
```
</answer>
"""

FINAL_CONTRACT = \
"""
You have exceeded the number of rounds available give a brief description of what 
you attempted, what worked, what didn't, and any additional information that would be required

> You *MUST* answer with the following format
```
<cot> Bried cot summary </cot>
<summary> Your summary </summary>
```
"""

GLOBAL_FORMAT = \
"""
You are an API‑facing language model.
Your responses will be consumed **programmatically**: after the caller strips the wrapper tags, the payload inside `<answer>` (or `<summary>`) must be *ready for `json.loads()`* or direct text use without further cleaning.

### 1 - General wrapper rules

1. Produce **exactly one** `<cot> … </cot>` block followed immediately by **exactly one** `<answer> … </answer>` block (or `<summary> … </summary>` for the final contract).
2. The `<cot>` block contains a **very brief** chain‑of‑thought (≤ 4 short sentences).
3. Nothing—not even whitespace—may appear before `<cot>` or after `</answer>` / `</summary>`.
4. **NEVER** emit Markdown fences, back‑ticks, “\`\`\`”, or language hints such as `json`.
5. **Do not escape quotation marks** inside JSON beyond normal JSON requirements; avoid `\"` unless the character is part of a string value.

### 2 — JSON hygiene checklist (for contracts that require JSON)

* Must be **valid JSON**: double‑quoted keys/strings, no comments, no trailing commas.
* Boolean literals are lower‑case `true` / `false`.
* Arrays may be pretty‑printed or minified; either is acceptable.
* **Do not nest `<cot>` or `<answer>` tags inside JSON values.**

Follow these instructions **exactly** so downstream code can parse your output without post‑processing.
"""
