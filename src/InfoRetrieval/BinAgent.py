import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()

@staticmethod  
def __create_system_prompt(clueweb_id: str) -> str:
    """
    Builds a single system prompt that injects the full article URL and text,
    instructing the LLM to return only a JSON object with bin_number and claim.

    Args:
        clueweb_id (str): The ClueWeb22-ID to search for.

    Returns:
        str: The complete system prompt ready to send as the LLM's system message.
    """
    corpus_path = os.path.join(
        'clueweb',
        'TREC-LR-2024',
        'T2',
        'trec-2024-lateral-reading-task2-baseline-documents.jsonl'
    )
    document = None

    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get('ClueWeb22-ID') == clueweb_id:
                    document = entry
                    break
    except FileNotFoundError:
        raise RuntimeError(f"Corpus file not found: {corpus_path}")

    if document is None:
        raise ValueError(f"Document ID {clueweb_id} not found in corpus")

    url = document.get('URL', '')
    full_text = document.get('Clean-Text', '').replace('\n', ' ').strip()

    system_prompt = f"""
                    You are a **Document‑Type Classification Agent**.

                    You have the following full context:

                    URL:
                    {url}

                    Full Content:
                    {full_text}

                    ——————————
                    Your task
                    ——————————

                    1. **Classify the article into exactly one of the following bins** (choose the single best fit):

                    0 = Political / Public-Policy News  
                    1 = Scientific / Research Article (peer-reviewed, preprint, or conference paper)  
                    2 = Health / Medical Guidance (consumer-facing medical or wellness advice)  
                    3 = Business / Economics & Finance (markets, company earnings, economic indicators)  
                    4 = Technology & Product Coverage (tech news, product reviews, security disclosures)  
                    5 = Investigative/Long-form Report (whistle-blowers, leaked documents, deep dives)  
                    6 = Opinion/Commentary — includes columns, op-eds **and pieces that advance an explicit moral / ethical value-judgement**.
                    7 = Data-Journalism / Explainer (data driven visualisations or explanatory deep dives)

                    2. **Extract the core claim and or theme** of the article in 1–2 sentences.  
                    • Preserve the author’s framing—do **not** neutralise or soften bias.  

                    ——————————
                    Output format
                    ——————————
                    Return your reply in **two consecutive blocks**:

                    <thinking> *Briefly explain your reasoning here* </thinking>  
                    <answer>  
                    {{  
                    "bin_number": <integer 0–8>,  
                    "claim": "<your extracted claim>"  
                    }}  
                    </answer>

                    *No additional text, markdown fences, or keys are allowed outside those tags.*
                    """
    return system_prompt

@staticmethod
def bin_document(client,clueweb_id: str) -> None:
    prompt = __create_system_prompt(clueweb_id)

    response = client.responses.create(
        model="o3-mini",
        input=prompt,
    )

    raw_reply = response.output_text
    print(f"Raw reply: {raw_reply}")

    # --- Extract the JSON between <answer> … </answer> ---
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw_reply, re.DOTALL)
    if not match:
        raise ValueError("No <answer> tag with JSON found in the model reply.")

    json_text = match.group(1).strip()

    # Optional: safe‑parse to ensure valid JSON
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inside <answer> is invalid: {e}")

    # Save pretty‑printed JSON
    os.makedirs(os.getenv("BIN_AGENT_PATH"), exist_ok=True)
    with open(os.getenv("BIN_AGENT_RESULTS_PATH"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    Dylan = "clueweb22-en0042-56-04016"
    Dan = "clueweb22-en0024-53-03398"
    Malaria = "clueweb22-en0019-43-01843"
    Oliwia = "clueweb22-en0030-87-05450"
    bin_document(Oliwia)
