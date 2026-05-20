import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load configuration keys and endpoints from the local environment (.env file)
load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

# Initialize our Azure OpenAI connection client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Start our conversation log with a system prompt that dictates the assistant's personality
messages_history=[
        {
            "role": "system",
            "content": "You are a helpful ,witty and concise AI assistant.",
        },
        
]


# Run a continuous loop to chat interactively with the user in the terminal
while True:
    user_input=input("\nUser: ")

    # Let the user exit the loop by typing exit, quit, or q
    if user_input.lower() in ["exit","quit","q"]:
        print("Exiting the chat")
        break

    # Record the user's question to the conversation history
    messages_history.append({"role":"user","content":user_input})

    print("\nAssistant: ",end="",flush=True)

    # Call Azure OpenAI to get a response, streaming the text back chunk-by-chunk
    response = client.chat.completions.create(
        stream=True,
        messages=messages_history,
        model=deployment
    )

    # Accumulate the streamed chunks and print them as they arrive
    full_response=""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response+=chunk.choices[0].delta.content
    
    print("\n")

    # Record the assistant's complete reply to the history so it remembers what it said
    messages_history.append({"role":"assistant","content":full_response})

# Close the API client connection nicely
client.close()