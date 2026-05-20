import os
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv


load_dotenv()

files=["remote_work_policy.txt","travel_expense_policy.txt","customer_data_handling_policy.txt"]

model=AzureChatOpenAI(
    model="gpt-4.1-mini",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

vector_store=InMemoryVectorStore(embeddings)

documents=[]
for file in files:
    loader=TextLoader(file)
    docs=loader.load()
    documents.extend(docs)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
all_split=text_splitter.split_documents(documents)

print(f"Split {len(all_split)} chunks")


document_id=vector_store.add_documents(all_split)

print(f"Added {len(document_id)} chunks to the vector store")
print(document_id[:3])

results = vector_store.similarity_search("what is work policy?")

for r in results:
    print(r.page_content)
    # print(r.metadata)

# print(results.page_content)