# Trec Ideas

## 1. Classify the document

Using article title and brief summary of the article, an LLM will classify the document based on the following bins:

| Bin   | Name                      | What it signals                                                  | Example                                                    |
| ----- | ------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **0** | **Verifiable Fact**       | Objective statements you can check against authoritative sources | “Water boils at 100 °C at sea level.”                      |
| **1** | **Statistical Claim**     | Quantitative or data-driven assertions requiring evidence        | “In Q1 2025, global unemployment fell to 5.2 %.”           |
| **2** | **Causal Claim**          | Cause-and-effect relationships—often backed by studies           | “Regular exercise reduces the risk of heart disease.”      |
| **3** | **Interpretive Claim**    | Analyses or contextual conclusions drawn from facts/data         | “Rising broadband access is driving remote-work adoption.” |
| **4** | **Value Judgment**        | Normative “ought to” statements reflecting moral/social values   | “Governments should ban single-use plastics.”              |
| **5** | **Subjective Preference** | Purely personal tastes or opinions with no objective grounding   | “Blue is the best color.”                                  |

## 2. Bin Specific Handling

### Bins 0-2 

These require minimal effort, using a search API or an LLMs internal context should easily be able to verify the claim of the document and provide a concise report.

### Bins 3-4

* Identify the core claim

Set up LLMs in a debate format. One agent supports the claim (has access to the full document), one agent refutes the claim (has access to the document), and one agent is the moderator. Throughout the conversation with these two agents the moderator (will not have acess to the document) will be generating the summary of both sides, and the moderator will have access to a Google search API to verify the claims made in the debate. At the end of the report the moderator will then give a closing remark on the claim.

> See more more details on information retrieval in the <code>Information Retrieval</code> section.

<p>

The moderator will essentially be doing a form of lateral reading by fact checking both sides of the debate.

### Bin 5

Provide a description of the claim and make a note that it is highly subjective.

# Information Retrieval

``` mermaid
flowchart LR
  UserQuery --> EmbedQuery
  EmbedQuery -->|similarity| VectorDB
  UserQuery -->|text search| KeywordIndex
  VectorDB --> CandidateDocs
  KeywordIndex --> CandidateDocs
  CandidateDocs --> RankAndFilter
  RankAndFilter -->|if score < T| FallbackWebSearch
  RankAndFilter --> FinalContext
  FinalContext --> LLM 
```

1. EmbedQuery: convert the user’s question into a vector (e.g. OpenAI embeddings).
2. VectorDB: search your local FAISS/Pinecone/Weaviate index for the top-K semantically closest docs.
3. KeywordIndex: run a BM25 (or Elasticsearch) query in parallel for exact term matches.
4. Merge & Rank: unify those two lists, dedupe, then rank by a combined score (e.g. α·cosine + (1−α)·BM25).
5. Threshold Check: if your best candidate’s score is below a set threshold, it likely means “I don’t really know this locally”—so fall back to a live web search API.
6. LLM Augmentation: feed the top N local docs (and/or web snippets if you fell back) into your prompt.

See more at [Here](https://chatgpt.com/share/68336792-7d48-8012-91b2-b471b190bcf7)