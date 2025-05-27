from openai import OpenAI
import os
import json

def main():
    client = OpenAI()

    # Load debate format (e.g., "ping_pong")
    debate_style = "ping_pong"
    format = json.load(open(f"debate_formats/{debate_style}/format.json"))
    

    # Store agent system prompts in a dictionary for quick access
    system_prompts = {
        agent["name"]: open(agent["system_prompt"]).read()
        for agent in format["agents"]
    }

    debate_log = []
    previous_response_id = None  # For conversation threading

    for step in format["sequence"]:
        agent_name = step["agent"]
        turn_prompt_path = step["turn_prompt"]
        user_prompt = open(turn_prompt_path).read()

        response = client.responses.create(
            model="gpt-4o",
            instructions=system_prompts[agent_name],
            input=[{"role": agent_name, "content": user_prompt}],
            previous_response_id = previous_response_id,  # Maintains context
            service_tier="flex",
        )

        # Save response content to debate log
        response_text = response.output[0].content
        debate_log.append({
            "agent": agent_name,
            "prompt": user_prompt,
            "response": response_text
        })

        # Update previous_response_id to maintain context
        previous_response_id = response.id

    # Print the full debate log
    print("\n=== Debate Log ===\n")
    for entry in debate_log:
        print(f"{entry['agent']} says:\n{entry['response']}\n")

if __name__ == "__main__":
    main()
