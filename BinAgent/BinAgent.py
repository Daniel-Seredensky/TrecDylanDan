import json
import os
from openai import OpenAI
from dotenv import load_dotenv

class BinAgent: 

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
                        You are a document classification agent.
                        You have the following full context:

                        URL:
                        {url}

                        Full Content:
                        {full_text}

                        Your task:
                        1. Extract the core claim from the article (1-2 sentences). It is very important that the claim is clear and will not lead to any misinterpretation. It is also very important that you do not alter the original bias of the text. As this claim will be the subject of debate.
                        2. Assign that extracted claim to exactly one of these bins:
                        0 = Verifiable Fact
                        1 = Statistical Claim
                        2 = Causal Claim
                        3 = Interpretive Claim
                        4 = Value Judgment
                        5 = Subjective Preference

                        Output
                        ------
                        Return exactly and only a JSON object with two fields. DO NOT WRAP YOUR ANSWER IN QUOTES OR TICKS:
                        {{
                        "bin_number": <integer 0–5>,
                        "claim": "<your identified claim>"
                        }}
                        """

        return system_prompt

    @staticmethod
    def bin_document(clueweb_id: str) -> None:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("API_KEY"))
        prompt = BinAgent.__create_system_prompt(clueweb_id)

        response = client.responses.create(
            model="o3-mini",
            input= prompt,
        )
        with open('BinAgent/DerivedData/Result.json', 'w') as f:
            text = response.output_text
            sub = text.replace("```json", "").replace("```", "")
            f.write(sub)
        print(f"Output:{response.output_text}")

if __name__ == "__main__":
    Dylan = "clueweb22-en0042-56-04016"
    Dan = "clueweb22-en0024-53-03398"
    Malaria = "clueweb22-en0019-43-01843"
    Oliwia = "clueweb22-en0030-87-05450"
    BinAgent.bin_document(Oliwia)

    load_dotenv()
    client = OpenAI(api_key=os.getenv("API_KEY"))
    prompt = create_system_prompt(Dan)

    response = client.responses.create(
        model="o3-mini",
        input=create_system_prompt(Dan),
    )
    with open('BinAgent/DerivedData/Result.json', 'w') as f:
        text = response.output_text
        sub = text.replace("```json", "").replace("```", "")
        f.write(sub)
    print(response.output_text)  # Print to console
