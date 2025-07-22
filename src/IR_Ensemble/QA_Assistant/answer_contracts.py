SEARCH_CONTRACT = \
"""
Given the following question and context relative to a topic document, return a json of bm25 optimized keyword 
queries (MARCO search) and a master query (used for semantic rerank). You may have up 
to *4* [queries,master_query] pairs in your "searches" array. As well as up to *4* queries per search, not including the master query.


> You *MUST* answer with the following format:
> Do **NOT** forget to close any tags or brackets.
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
        "master_query": <master_query>
        },
        ...
    ] 
} # Do not forget to close curly brackets
</answer>
```
"""

SELECT_CONTRACT = \
"""
Given the previou questions, topic context, and the search result metadata choose the most promising sources to answer the question.
Select up to *6* segment_ids for further exploration

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

UPDATE_CONTRACT = \
"""
You are a **Information Retrieval Assistant** -- updating an answer to a question.
Given the previous context and the search results given below update your answer status
*DO NOT* remove any existing citations, but you may add new ones.
Immediately upon marking a question as true it will be removed from the next round.
> Since this is a fact checking assignment the document context is any relevant information from the document we are fact checking that you may need in your answer.
> Do **NOT** cite anything other than a Marco segment id, leave blank citations array if no citations exist.


> You *MUST* answer with the following format
```
<cot> Brief cot summary</cot> 
<answer>
{
"questions": [
        {
            "question": <verbatim user question>,
            "doc_context": <verbatim doc context>,
            "answer": 
                {
                    "text": <text>,
                    "citations": [
                        {
                            "summary": <summarize the info used from the citation>,
                            "citation": <segment_id>,
                        },
                        ...
                    ]
                }
            "finished": <true if fully confident and finished working, false otherwise>
        },
        ...
    ],
"rounds": [
        {
            "summary": <Brief summary of the round and different kw queries you tried that did not yield results to avoid in the future>,
            "seen_ids": [ # A list of seen search results that way they can be avoided in the future
                <segment_id1>,
                <segment_id2>,
                ...
            ]
        },
        {
            "summary": <Next rounds summary>,
            "seen_ids": [ # *ONLY* the seen ids from this round no repeat ids from previous rounds
                <segment_id3>,
                <segment_id4>,
                ...
            ]
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
2. The `<cot>` block contains a ** brief** chain‑of‑thought ).
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

PLAN_NEXT_CONTRACT = \
"""
You are a **Information Retrieval Planner** -- creating a plan for the upcoming round of a question answering system.
A full round looks like this:
Plan → Search (queries and master query) → Select (SEGMENT_ID, from metadata) → Update Answer (GIVEN CONTEXT) -> Next round starts back up with Plan

Below you will be given the following information:

* `current_answer` – This is a json object, what will be relevant to you are the questions with document context (more about this later), current answers, and round summaries. Use these to guide your planning for the upcoming round.
* `max_rounds` – the total round budget
* `current_round` – the current round
* `questions` – a list of questions and document context
> Since this is a fact checking assignment the document context is any relevant information from the document we are fact checking that you may need in your answer.

> If this is the first round, you will be given `questions` instead of `current_answer` but this does not alter your goal.

Using this information you must craft **one** focused plan for the **upcoming** round only.  
Base the plan on what is still missing, lessons in previous `rounds[i].summary`, and any `seen_ids` to avoid duplicates.
Try to craft a plan with 1-2 focused searches each search contains max 4 queries and a master query for semantic rerank, avoid using similar queries as previous round summaries.

---

> You *MUST* answer with the following format
```
<cot> Brief cot summary</cot> 
<answer>
Your plan for the next round
</answer>
```
"""