import requests
from pathlib import Path

def run_debate_pipeline(context_path: str = "DerivedData/ContextBuilder/Context.txt", debate_api_url: str = "http://127.0.0.1:7860/api/v1/run/ff071340-7374-4b9b-b222-bb157ff5be7b") -> str:
    """
    Reads the context document and sends it to the debate pipeline API. Returns the debate summary (API response text).
    Args:
        context_path: Path to the context document (default: DerivedData/ContextBuilder/Context.txt)
        debate_api_url: Debate pipeline API endpoint
    Returns:
        The debate summary as a string (or error message)
    """
    context_file = Path(context_path)
    if not context_file.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    context = context_file.read_text(encoding="utf-8")
    payload = {
        "input_value": context,
        "output_type": "text",
        "input_type": "text"
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(debate_api_url, json=payload, headers=headers)
        response.raise_for_status()
        # Try to extract the debate summary from the response
        data = response.json()
        # Try to extract the text from the nested structure if present
        try:
            return data["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"]
        except Exception:
            # Fallback: return the whole response as string
            return str(data)
    except Exception as e:
        return f"Error during debate pipeline call: {e}" 