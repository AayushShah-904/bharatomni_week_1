import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

sample_transcript = """
I am having trouble logging into my account. I keep getting an error message that says
'Invalid credentials' even though I am sure I am entering the correct password.
Can you help me resolve this issue?
"""

schema = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "account", "other"]
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        },
        "summary": {
            "type": "string"
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"}
        },
        "suggested_next_step": {
            "type": "string"
        }
    },
    "required": [
        "category",
        "urgency",
        "summary",
        "missing_information",
        "suggested_next_step"
    ],
    "additionalProperties": False
}

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "system",
            "content": "Classify customer support messages and return only valid JSON."
        },
        {
            "role": "user",
            "content": f"""
                    Analyze this support message and return JSON.

                    Support message:
                    {sample_transcript}
                    """
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "support_ticket",
            "strict": True,
            "schema": schema
        }
    },
)

def validate_ticket(data):
    required = {
        "category", "urgency", "summary",
        "missing_information", "suggested_next_step"
    }

    if not required.issubset(data.keys()):
        return False, "Missing required fields"

    if data["category"] not in {"billing", "technical", "account", "other"}:
        return False, "Invalid category"

    if data["urgency"] not in {"low", "medium", "high"}:
        return False, "Invalid urgency"

    if not isinstance(data["missing_information"], list):
        return False, "missing_information must be a list"

    return True, "Valid"

content = response.choices[0].message.content
data = json.loads(content)

is_valid, message = validate_ticket(data)
print(f"Validation result: {message}")
print(json.dumps(data, indent=2))
client.close()








#valideata input and output from pyandianitc
# quato
# api server
# half way server crash