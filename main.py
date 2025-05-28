from openai import OpenAI
import os
import json

def main():
    client = OpenAI()

    # Load debate format (e.g., "ping_pong")
    debate_style = "ping_pong"
    format_path = f"debate_formats/{debate_style}/format.json"
    format = json.load(open(format_path))

    # Load the claim from the bin agent data
    with open("BinAgent/DerivedData/Result.json", 'r', encoding='utf-8') as f:
        claim_data = json.load(f)
    claim = claim_data["claim"]
    clueweb_id = claim_data.get("ClueWeb22-ID", "clueweb22-en0024-53-03398")  # fallback if not included

    # Load the article from the corpus
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

    # Prepare system prompts for each agent
    system_prompts = {
        agent["name"]: open(
            f"debate_formats/{debate_style}/prompts/{agent['system_prompt']}"
        ).read()
        for agent in format["agents"]
    }

    debate_log = []
    previous_response_id = None  # For conversation threading

    for step in format["sequence"]:
        agent_name = step["agent"]
        turn_prompt_path = f"debate_formats/{debate_style}/{step['turn_prompt']}"
        base_prompt = open(turn_prompt_path).read().strip()

        # Prepare the claim section
        claim_section = f"\n\n[CLAIM TO BE DEBATED]\n{claim}"

        # Provide ~5 sentences from article for all agents
        sentences = full_text.split('. ')
        snippet = '. '.join(sentences[:5]).strip()
        if not snippet.endswith('.'):
            snippet += '.'

        context_section = f"\n\n[EXCERPT FROM ARTICLE]\n{snippet}"

        relevant_articles = open("ContextForTesting/Context.jsonl").read()
        context_section += "OTHER RELEVANT ARTICLES TO CITE: " + relevant_articles

        # Final user prompt
        user_prompt = f"{base_prompt}{claim_section}{context_section}\n\nBased on the excerpt above, respond accordingly."

        response = client.responses.create(
            model="o3-mini",
            instructions=system_prompts[agent_name],
            input=[{"role": "user", "content": user_prompt}],
            previous_response_id=previous_response_id,
        )

        # Save to debate log
        response_text = response.output[1].content[0].text
        debate_log.append({
            "agent": agent_name,
            "prompt": user_prompt,
            "response": response_text
        })

        previous_response_id = response.id

    # Export debate log to text file
    with open("debate_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write("=== Debate Log ===\n\n")
        for entry in debate_log:
            log_file.write(f"{entry['agent']} says:\n{entry['response']}\n\n")

    article_json_str = json.dumps(document, indent=2)

    response = client.responses.create(
        model="o3-mini",
        input=[{
            "role": "user",
            "content": open("report_prompt.txt").read()
            + "\n DEBATE: " + open("debate_log.txt").read()
            + "\n ARTICLE: " + article_json_str
            + "\n OTHER RELEVANT ARTICLES: " + relevant_articles
        }],
    )

    print(response.output[1].content[0].text)

    response = client.responses.create(
        model="o3-mini",
        input=[{
            "role": "user",
            "content": open("report_prompt.txt").read()
            + "\n ARTICLE: " + article_json_str
            + "\n OTHER RELEVANT ARTICLES: " + relevant_articles
        }],
    )

    print(response.output[1].content[0].text)


if __name__ == "__main__":
    main()
