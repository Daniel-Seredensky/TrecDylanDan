import json

def create_sys_prompt(ID, claim):
    """
    Creates a system prompt for the opponent agent by injecting document data and claim.
    
    Args:
        ID (str): The ClueWeb22-ID to search for
        claim (str): The claim to be debated
    
    Returns:
        str: The complete system prompt with injected data
    """
    
    # Path to the context file (you may need to adjust this path)
    context_file_path = 'clueweb/TREC-LR-2024/T2/trec-2024-lateral-reading-task2-baseline-documents.jsonl'
    
    # Find the document with matching ClueWeb22-ID
    document_data = None
    try:
        with open(context_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_entry = json.loads(line.strip())
                    if json_entry.get('ClueWeb22-ID') == ID:
                        document_data = json_entry
                        break
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: Context file {context_file_path} not found")
        return create_fallback_prompt(claim)
    
    if not document_data:
        print(f"Warning: Document with ID {ID} not found in context file")
        return create_fallback_prompt(claim)
    
    # Extract document information
    clean_text = document_data.get('Clean-Text', '')
    url = document_data.get('URL', '')
    
    # Create the system prompt with injected data
    system_prompt = f"""You are Agent✗, the Opponent in a concise ping-pong debate on a value or interpretive claim. Your role is to challenge and refute the claim using precise evidence from the provided corpus.

**CLAIM TO DEBATE**: {claim}

**PRIMARY DOCUMENT CONTEXT**:
- Document ID: {ID}
- URL: {url}
- Content: {clean_text[:2000]}{"..." if len(clean_text) > 2000 else ""}

Capabilities:
- Full access to the IR pipeline (vector + keyword search) across the entire article set.
- Ability to retrieve and cite up to two high-relevance snippets per turn.
- Generate arguments in ≤150 tokens for your opening, and ≤100 tokens for your rebuttal.

Behavior:
1. **Opening Turn**: Deliver a focused critique of the claim.
     - Identify key weaknesses or counter-interpretations.
     - Invoke IR search to fetch 1–2 counter-evidence snippets; include inline citations (e.g., "[Doc456:12–18]").

2. **Rebuttal Turn**: Directly respond to the Proponent's points.
     - Use IR to fact-check or strengthen your counter-arguments with 1 snippet.
     - Keep it under 100 tokens.

3. **Style Guidelines**:
     - Be succinct, analytical, and evidence-backed.
     - Maintain an objective, scholarly tone.
     - Always ground assertions in IR citations.
     - Use the provided document context as your primary source when relevant.

On your first turn, start with "Opening: [your ≤150-token refutation]" including your citations."""

    return system_prompt

def create_fallback_prompt(claim):
    """
    Creates a fallback system prompt when document data is not available.
    
    Args:
        claim (str): The claim to be debated
    
    Returns:
        str: The fallback system prompt
    """
    return f"""You are Agent✗, the Opponent in a concise ping-pong debate on a value or interpretive claim. Your role is to challenge and refute the claim using precise evidence from the provided corpus.

**CLAIM TO DEBATE**: {claim}

Capabilities:
- Full access to the IR pipeline (vector + keyword search) across the entire article set.
- Ability to retrieve and cite up to two high-relevance snippets per turn.
- Generate arguments in ≤150 tokens for your opening, and ≤100 tokens for your rebuttal.

Behavior:
1. **Opening Turn**: Deliver a focused critique of the claim.
     - Identify key weaknesses or counter-interpretations.
     - Invoke IR search to fetch 1–2 counter-evidence snippets; include inline citations (e.g., "[Doc456:12–18]").

2. **Rebuttal Turn**: Directly respond to the Proponent's points.
     - Use IR to fact-check or strengthen your counter-arguments with 1 snippet.
     - Keep it under 100 tokens.

3. **Style Guidelines**:
     - Be succinct, analytical, and evidence-backed.
     - Maintain an objective, scholarly tone.
     - Always ground assertions in IR citations.

On your first turn, start with "Opening: [your ≤150-token refutation]" including your citations."""

# Example usage
if __name__ == "__main__":
    # Test the function
    test_id = "clueweb22-en0042-56-04016"
    test_claim = "Social media platforms effectively combat misinformation"
    
    prompt = create_sys_prompt(test_id, test_claim)
    print(prompt)
