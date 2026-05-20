import os
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv


load_dotenv()

# 1. Load the policy text files that we want our AI to read and answer questions from
files = ["remote_work_policy.txt", "travel_expense_policy.txt", "customer_data_handling_policy.txt"]

# 2. Configure our AI model using Azure OpenAI
model = AzureChatOpenAI(
    model="gpt-4.1-mini",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# 3. Set up the embedding model to convert text chunks into vector numbers
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# 4. Create an in-memory database to store our document pieces
vector_store = InMemoryVectorStore(embeddings)

# 5. Read the content of each policy file
documents = []
for file in files:
    loader = TextLoader(file)
    docs = loader.load()
    documents.extend(docs)

# 6. Split the long policy documents into smaller, bite-sized text chunks (1,000 characters each)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_split = text_splitter.split_documents(documents)

print(f"Split {len(all_split)} chunks")

# 7. Add these text chunks into our vector database
document_id = vector_store.add_documents(all_split)

print(f"Added {len(document_id)} chunks to the vector store")
print(document_id[:3])

# 8. Ask a question and retrieve the most relevant chunks from our database
results = vector_store.similarity_search("what is work policy?")

# 9. Print out the search results we found
for r in results:
    print(r.page_content)
    # print(r.metadata)

# print(results.page_content)