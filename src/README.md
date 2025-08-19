# Information Retrieval

The core of any RAG (Retrieval-Augmented Generation) system is its information retrieval process. The quality of your results depends on the data you can access. Below is an overview of how this system is designed.

## Overview

The MARCO V2.1 segmented dataset is very large, and our compute resources are limited. Embedding the entire dataset in a vector store isn’t feasible, so we’ve designed a hybrid retrieval pipeline that balances **recall** (finding as many relevant passages as possible) with **cost** and **latency** (minimizing expensive model calls).

1. **Query Generation**

   * **What happens:** An LLM produces 2–8 BM25‑optimized queries plus a separate “master” query.
   * **Why:** Multiple, targeted BM25 queries boost recall by covering different phrasings; the master query serves as the anchor for later refinement.

2. **Synonym Expansion & BM25 Search**

   * **What happens:**

     1. Each BM25 query is expanded via a synonym map to capture lexical variants.
     2. All expanded queries run **in parallel** against a local Lucene index.
     3. **Results are collapsed on `document_id`** to remove duplicate segments produced by the dataset’s sliding‑window segmentation.
     4. We fuse the collapsed results using Reciprocal Rank Fusion (RRF).
   * **Why:**

     * **Cost:** Local BM25 searches and synonym expansion are very cheap.
     * **Recall:** Collapsing on `document_id` prevents over‑weighting duplicated segments, while RRF fusion amalgamates signals across queries.

3. **Cohere Cross‑Encoder Reranking**

   * **What happens:**

     1. We send the PRF‑augmented master query along with the **top 600** RRF results to the Cohere Cross‑Encoder (CE).
     2. The CE reranks and returns **the top 75** document metadata.
   * **Why:**

     * **Recall vs. Cost Trade‑off:**

       * Using 600 candidates maximizes recall by giving the CE a wide pool to reorder.
       * Restricting to 600 (instead of, say, 1,000+) keeps CE costs and latency within reasonable bounds.
     * **Simplicity:** A single CE call is straightforward to implement and monitor.

4. **LLM Context & Document Selection**

   * **What happens:** The top 75 metadata records (title, URL, headers, snippet IDs) are provided to the LLM. It then selects which documents (or segments) to read in full.
   * **Why:**

     * **Cost Efficiency:** Passing only 75 candidates to the LLM keeps context lengths manageable and API usage low.
     * **Precision:** The final selection ensures the LLM works with the most promising documents.

5. **Autonomous Question Evaluation**

   * **What happens:** Through the Azure Assistants API, the LLM iteratively composes an answer to the original question, drawing on the selected full‑text documents with minimal human oversight.
   * **Why:**

     * **Scalability:** Automates the Q\&A loop so the system can handle many queries in parallel.
     * **Quality:** The structured retrieval steps feed high‑quality evidence into the answer process.

---

By collapsing on `document_id`, using cheap local BM25 + RRF for broad coverage, applying low‑cost PRF for targeted refinement, and then investing in a single, bounded CE call (600 → 75), this pipeline maximizes recall while keeping model‑inference costs and overall latency under control.

## File Structure

> Wherever possible, asynchronous operations are used to improve performance when multiple agents share a thread.

``` plaintext

src
└── InfoRetrieval
├   ├── lib                 # Dependency JAR files
├   ├── BatchManager.py     # Push/pull to Azure AI Search Index
├   ├── QuestionEval.py     # Question Evaluation Agent implementation
├   ├── BinAgent.py         # (Deprecated) Document-based templates are in development
├   ├── Utils.py            # Utility functions
├   ├── JsonIndexer.java    # Converts raw JSON data into a Lucene index
├   └── Search.java         # Performs initial parallel searches on the local index
└── ContextBuilder.py       # Main driver for the information retrieval process

```
 
## More Details

### BatchManager

`BatchManager` is a singleton class with these responsibilities:
- Prevent duplicate documents from being sent to the Azure AI Search index  
- Embed documents before uploading to the index  
- Asynchronously track the upload status so the LLM can run hybrid queries as soon as documents are ready  
- Asynchronously retrieve search results from the index

### QuestionEval

`QuestionEval` is the workhorse of the RAG system. Using Azure’s Assistants API, a Question Evaluation Agent can:
- Take a topic and corresponding question  
- Use local information retrieval tools to collect evidence  
- Iteratively refine its answer until it reaches confidence  

Multiple Question Evaluation Agents can process a full document of questions in parallel. Their answers are aggregated to create a comprehensive context for Debate Agents.

#### Agent Flow Diagram

``` mermaid

flowchart TD
    A["Start Agent Lifecycle"]
    B["Initialize QuestionAssessmentAgent"]
    C["Create Assistant (async)"]
    D["Create Thread"]
    E["Send User Question"]
    F["Kick-off Assistant Run"]
    G["Start Polling Loop<br>(iteration < MAX_ITERATIONS)"]
    H{"run.status?"}
    I["Sleep POLL_INTERVAL_SECONDS"]
    K1["Get Latest Message & Update Status<br>(on completed)"]
    K2["Get Latest Message & Update Status<br>(on requires_action)"]
    L["Return {status, content}"]
    M["Dispatch Tool Calls"]
    N["Submit Tool Outputs"]
    Q["Unexpected State → Error"]
    R["Max Iterations Exceeded"]
    S["Send Fallback User Message"]
    T["Create Short Follow-up Run"]
    U{"Short Run Status Completed?"}
    V["Get Latest Message & Update Status<br>(on short run)"]
    W["Return Last Status & Content"]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> |"queued / in_progress"| I --> H
    H --> |"completed"| K1 --> L
    H --> |"requires_action"| M --> N --> K2 --> G
    H --> |"other"| Q

    G --> |"iteration == MAX_ITERATIONS"| R --> S --> T --> U
    U --> |"yes"| V --> L
    U --> |"no"| W

``` 

#### Search Flow Diagram

``` mermaid

flowchart TD
    A["Start: Receive Question"]
    B["LLM Generate BM25 Queries (6–12) & Seed Paragraph"]
    C["Parallel BM25 Search for Each Query"]
    D["RRF Rerank All Search Results"]
    E["Select Top 200 Reranked Documents"]
    F["Embed Top 200 & Index in Azure AI Search"]
    G["Hybrid Search: BM25 Queries + Seed Paragraph"]
    H["Retrieve Top 75 Document Metadata"]
    I["Pass Top 75 Metadata to LLM as Context"]
    J["LLM Document Selection (choose most relevant)"]
    K["Selected Documents → Detailed Reading"]
    L["End"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L

```

### Java Components

**Compilation**

``` bash

javac -cp "src/IR_Ensemble/QA_Assistant/Search/lib/*:." src/IR_Ensemble/QA_Assistant/Search/*.java

```

**Execution**

``` bash
java -cp "src/IR_Ensemble/QA_Assistant/Search/lib/*:." src.IR_Ensemble.QA_Assistant.Search.DocumentSelection
java -cp "src/IR_Ensemble/QA_Assistant/Search/lib/*:." src.IR_Ensemble.QA_Assistant.Search.Searcher
```

# Current plans
