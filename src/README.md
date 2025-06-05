# Information Retrieval

The core of any RAG (Retrieval-Augmented Generation) system is its information retrieval process. The quality of your results depends on the data you can access. Below is an overview of how this system is designed.

## Overview

The dataset for this project is very large, and our compute resources are limited. Embedding the entire dataset in a vector database is not feasible. To overcome this, we use a hybrid approach:

1. **Query Generation**  
   For each question, an LLM generates:
   - A list of 6–12 BM25-optimized queries  
   - A seed paragraph for semantic search

2. **BM25 Search & Reranking**  
   - Run the 6–12 queries in parallel against a local Lucene index  
   - Rerank all results using RRF (Ranked Relevance Feedback)  
   - Select the top 200 reranked documents

3. **Embedding & Azure Indexing**  
   - Embed the top 200 documents  
   - Upload the embeddings to an Azure AI Search index

4. **Hybrid Search**  
   - Search the Azure index using the original BM25 queries and the seed paragraph  
   - Retrieve the top 75 document metadata

5. **LLM Context & Document Selection**  
   - Pass the top 75 metadata records to the LLM as context  
   - The LLM chooses the most relevant documents to read in full

6. **Autonomous Question Evaluation**  
   - Using the Azure Assistants API, the LLM iteratively answers the question with minimal oversight, leveraging the retrieved documents.

## File Structure

> Wherever possible, asynchronous operations are used to improve performance when multiple agents share a thread.

``` plaintext

src
└── InfoRetrieval
|   ├── lib                 # Dependency JAR files
|  ├── BatchManager.py     # Push/pull to Azure AI Search Index
|  ├── QuestionEval.py     # Question Evaluation Agent implementation
|  ├── BinAgent.py         # (Deprecated) Document-based templates are in development
|  ├── Utils.py            # Utility functions
|  ├── JsonIndexer.java    # Converts raw JSON data into a Lucene index
|  └── Search.java         # Performs initial parallel searches on the local index
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

javac -cp "src/InfoRetrieval/lib/*:." src/InfoRetrieval/*.java

```

**Execution**

``` bash

java -cp "src/InfoRetrieval/lib/*:." src.InfoRetrieval.JsonIndexer
java -cp "src/InfoRetrieval/lib/*:." src.InfoRetrieval.Search

```
