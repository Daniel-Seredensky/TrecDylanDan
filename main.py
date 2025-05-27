from openai import OpenAI
import os

client = OpenAI(
    api_key=open("api_key.txt").read(),
)

def main():
    response = client.responses.create(model="gpt-4o",
    instructions="Simply respond with 'Success'",
    input="Simply respond with 'Success'",
    )
    
    print(response.output_text)

if __name__ == "__main__":
    main()