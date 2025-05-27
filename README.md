# Trec Ideas

## Main Idea

### 1. Classify the document

Using article title and brief summary of the article, an LLM will classify the document based on the following bins:

| Bin   | Name                      | What it signals                                                  | Example                                                    |
| ----- | ------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **0** | **Verifiable Fact**       | Objective statements you can check against authoritative sources | “Water boils at 100 °C at sea level.”                      |
| **1** | **Statistical Claim**     | Quantitative or data-driven assertions requiring evidence        | “In Q1 2025, global unemployment fell to 5.2 %.”           |
| **2** | **Causal Claim**          | Cause-and-effect relationships—often backed by studies           | “Regular exercise reduces the risk of heart disease.”      |
| **3** | **Interpretive Claim**    | Analyses or contextual conclusions drawn from facts/data         | “Rising broadband access is driving remote-work adoption.” |
| **4** | **Value Judgment**        | Normative “ought to” statements reflecting moral/social values   | “Governments should ban single-use plastics.”              |
| **5** | **Subjective Preference** | Purely personal tastes or opinions with no objective grounding   | “Blue is the best color.”                                  |

### 2. Bin Specific Handling

#### Bins 0-2 

These require minimal effort, using a search API or an LLMs internal context should easily be able to verify the claim of the document and provide a concise report.

#### Bins 3-4

* Identify the core claim
Set up AI agents in debate format (see <code>Debate Format</code> below). At the end of the debate the moderator will generate a context summary based on the arguments made by the agents. Throughout the debate the moderator will also be able to fact check arguments made by the agents (see <code>Information Retrieval</code> below).

The moderator will essentially be doing a form of <b>lateral reading</b> by fact checking both sides of the debate.

#### Bin 5

Provide a description of the claim and make a note that it is highly subjective.

## Information Retrieval

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
2. VectorDB: search your local FAISS/Pinecone/Weaviate (clueweb for us) index for the top-K semantically closest docs.
3. KeywordIndex: run a BM25 (or Elasticsearch) query in parallel for exact term matches.
4. Merge & Rank: unify those two lists, dedupe, then rank by a combined score (e.g. α·cosine + (1−α)·BM25).
5. Threshold Check: if your best candidate’s score is below a set threshold, it likely means “I don’t really know this locally”—so fall back to a live web search API.
6. LLM Augmentation: feed the top N local docs (and/or web snippets if you fell back) into your prompt.

See more [Here](https://chatgpt.com/share/68336792-7d48-8012-91b2-b471b190bcf7)

## Possible Debate Formats

> REMOVE MODERATOR 

### 1. Mini Oxford-Style LLM Debate (≈7 turns)

* **Roles**

  * **Proponent (Agent✓)**—supports the interpretive/value claim.
  * **Opponent (Agent✗)**—refutes the claim.
  * **Moderator (Mod)**—introduces the claim, triggers retrieval checks, summarizes.

* **Turn Structure**

  1. **Mod:** “Claim: …“ (1 turn)
  2. **Agent✓:** Opening (≤200 tokens)
  3. **Agent✗:** Opening (≤200 tokens)
  4. **Mod:** “Fact-check Round”

     * Runs your Vector+BM25 merge on each side’s top 2 assertions.
     * Shares 1–2 vetted counter-evidence points (≤100 tokens).
  5. **Agent✓:** Rebuttal (≤150 tokens)
  6. **Agent✗:** Rebuttal (≤150 tokens)
  7. **Mod:** Closing summary (≤100 tokens) & verdict note

* **Why it works:**

  * Keeps the feed-forward flow of Oxford style.
  * Embeds a single, focused IR check mid-debate.
  * Fits neatly into a handful of LLM turns.

---

### 2. Four-Corners LLM Debate (≈5 turns)

* **Roles**

  * **Agent▲ (Agree)** & **Agent△ (Strongly Agree)**
  * **Agent▽ (Disagree)** & **Agent▼ (Strongly Disagree)**
  * **Mod** (same as above)

* **Turn Structure**

  1. **Mod:** “Claim: …“ (1)
  2. **Corner Statements:** Each of the four “corner” agents posts a **short** position (≤100 tokens each)
  3. **Mod:** IR check—pull top counter-examples for 2 corners with highest BM25 scores (1 turn)
  4. **Pair Rebuttals:** Two rounds where opposite-corner agents each respond (≤100 tokens each, 4 messages total)
  5. **Mod:** “Key takeaways” summary (≤150 tokens)

* **Why it works:**

  * Simulates group-think diversity in four viewpoints.
  * Moderator intervenes once to “lateral read” and disrupt echo chambers.
  * Very compact: ideal for a single LLM API session.

---

### 3. Micro Cambridge-Style LLM Debate (≈9 turns)

* **Roles**

  * **PM (Prime Minister)** & **LO (Leader of Opposition)**
  * **DPM (Deputy PM)** & **DLO (Deputy LO)**
  * **Mod**

* **Turn Structure**

  1. **Mod:** “Motion: …“ (1)
  2. **PM:** Opening (≤150 tokens)
  3. **LO:** Opening (≤150 tokens)
  4. **Mod:** Quick IR check on one key PM + one key LO point (1)
  5. **DPM:** Rebuttal (≤100 tokens)
  6. **DLO:** Rebuttal (≤100 tokens)
  7. **PM:** Reply (≤80 tokens)
  8. **LO:** Reply (≤80 tokens)
  9. **Mod:** Final summary & graded confidence (≤100 tokens)

* **Why it works:**

  * Offers layered openings and replies, teaching LLMs to structure argument depth.
  * Embeds exactly one moderation check to keep token-cost low.

---

### 4. Ping-Pong LLM Debate (≈6 turns)

* **Roles**

  * **Agent✓** vs. **Agent✗** vs. **Mod**

* **Turn Structure**

  1. **Mod:** “Prompt: …“ (1)
  2. **Agent✓:** Pro stance (≤150 tokens)
  3. **Agent✗:** Con stance (≤150 tokens)
  4. **Mod:** IR jump—fetch 1 supporting snippet for each side (1)
  5. **Agent✓:** Rebuttal (≤100 tokens)
  6. **Agent✗:** Rebuttal (≤100 tokens)
  7. *(Optional)* **Mod:** Verdict note (≤80 tokens)

* **Why it works:**

  * Ultra-concise “exchange.”
  * Moderator interjects once mid-pong to ground claims in real evidence.


See more [Here](https://chatgpt.com/share/68336e3b-dc3c-8012-99b2-8d7ceb9027c2)

---

Exact token counts may be adjusted as needed