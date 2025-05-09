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

These require minimal effort, using a search API an LLM should easily be able to verify the claim of the document and provide a concise report.

### Bins 3-4

* Identify the core claim

Set up LLMs in a debate format. One agent supports the claim, one agent refutes the claim, and one agent is the moderator. Throughout the conversation with these two agents the moderator will be generating the summary of both sides, and the moderator will have access to a Google search API to verify the claims made in the debate. At the end of the report the moderator will then give a closing remark on the claim.

### Bin 5

Provide a description of the claim and note that it is subjective.