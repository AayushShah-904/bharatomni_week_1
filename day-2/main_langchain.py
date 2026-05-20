import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/"
deployment = "gpt-4.1"
api_version = "2024-12-01-preview"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Setting up our LangChain Chat Model connected to Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=1.0,
    top_p=1.0,
    max_tokens=13107,
)


# Define the conversation starter messages: a system role instruction and a user question
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="how it will be if i make a rag based application where it takes input of github repo link and summarized the whole project and help the user to understand the project")
]

# We can invoke the model all at once (uncomment to test):
# response = llm.invoke(messages)
# print(response.content)

# Or we can stream the response chunk-by-chunk for a smoother, real-time feel:
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)