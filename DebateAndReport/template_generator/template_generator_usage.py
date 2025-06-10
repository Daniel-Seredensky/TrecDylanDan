import requests
import json

url = "http://127.0.0.1:7860/api/v1/run/d291fdb7-25cf-463d-a43e-a1a08e2ace49"  # The complete API endpoint URL for this flow

# Request payload configuration
payload = {
    "input_value": open("template_generator/article.txt", "r").read(),  # The input value to be processed by the flow
    "output_type": "text",  # Specifies the expected output format
    "input_type": "text"  # Specifies the input format
}

# Request headers
headers = {
    "Content-Type": "application/json"
}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    #open("template_generator/output.txt", "w").write(json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"])
    open("template_generator/output_1.json", "w").write(json.loads(response.text)["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"])
    open("template_generator/output_2.json", "w").write(json.loads(response.text)["outputs"][0]["outputs"][1]["results"]["text"]["data"]["text"])
    open("template_generator/output_3.json", "w").write(json.loads(response.text)["outputs"][0]["outputs"][2]["results"]["text"]["data"]["text"])

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")
    