import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/"
deployment = "gpt-4.1"
api_version = "2024-12-01-preview"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=1.0,
    top_p=1.0,
    max_tokens=13107,
)


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="how it will be if i make a rag based application where it takes input of github repo link and summarized the whole project and help the user to understand the project")
]

response = llm.invoke(messages)

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
# print(response.content)