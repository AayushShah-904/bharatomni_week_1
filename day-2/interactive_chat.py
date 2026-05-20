import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

messages_history=[
        {
            "role": "system",
            "content": "You are a helpful ,witty and concise AI assistant.",
        },
        
]


while True:
    user_input=input("\nUser: ")

    if user_input.lower() in ["exit","quit","q"]:
        print("Exiting the chat")
        break

    messages_history.append({"role":"user","content":user_input})

    print("\nAssistant: ",end="",flush=True)

    response = client.chat.completions.create(
        stream=True,
        messages=messages_history,
        model=deployment
    )

    full_response=""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response+=chunk.choices[0].delta.content
    
    print("\n")

    messages_history.append({"role":"assistant","content":full_response})

client.close()