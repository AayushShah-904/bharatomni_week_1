# This file runs the Q&A engine that searches our database
# and answers the user's questions about the codebase.

import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# Basic settings for the Q&A assistant

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "github_repo"
TOP_K           = 8    # chunks returned to LLM
MAX_HISTORY     = 6    # conversation turn pairs kept in context


# The instructions we give to the AI to define its behavior and tone

SYSTEM_PROMPT = """\
You are an expert software engineer assistant. Your job is to help users deeply \
understand a GitHub codebase. Answer the user's question based on the provided code context. \
If the context is insufficient to answer, say so clearly — never hallucinate code. \
Use markdown. Keep answers focused.
"""


# Helper functions for formatting search results and building conversation history

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        path = doc.metadata.get("source", "unknown")
        lang = doc.metadata.get("language", "")
        parts.append(f"### `{path}`\n```{lang}\n{doc.page_content}\n```")
    return "\n\n".join(parts)
def root():
    print("hello from react app.")
    pass

def _build_messages(
    question: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    """
    Combines the system prompt, chat history, and the newly retrieved code chunks
    into a structured list of messages that the AI model can understand.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Windowed conversation history
    messages.extend(history)

    # Chunks + question
    messages.append({
        "role": "user",
        "content": (
            f"## Relevant code from the repository\n\n{context}\n\n"
            f"---\n\n"
            f"**Question:** {question}"
        ),
    })

    return messages


# Main Q&A Class

class RepoQA:
    """
    This class handles searching the database and responding to user questions.
    It keeps track of the conversation history to make follow-up questions work smoothly.
    """

    def __init__(self, vectorstore: Chroma, repo_url: str = "") -> None:
        self.vs       = vectorstore
        self.repo_url = repo_url

        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            streaming=True,
        )

        # Holds the history of the conversation as a list of user/assistant messages
        self.history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        return self.history[-(MAX_HISTORY * 2):]

    def _record_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

    def stream_answer(self, question: str) -> Iterator[str]:
        """Finds relevant code chunks, streams the answer in real-time, and cites the sources."""
        docs        = self.vs.similarity_search(question, k=TOP_K)
        context     = _format_docs(docs)
        messages    = _build_messages(
            question, context, self._trim_history()
        )

        full_response = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        sources = sorted({
            d.metadata.get("source", "")
            for d in docs
            if d.metadata.get("source") not in ("", "__directory_tree__")
        })
        if sources:
            citation_block = "\n\n---\n**Sources:**\n" + "\n".join(f"- `{s}`" for s in sources)
            yield citation_block
            full_response += citation_block

        self._record_turn(question, full_response)

    def answer(self, question: str) -> tuple[str, list[Document]]:
        """Finds relevant code chunks and returns the complete answer and source documents at once."""
        docs        = self.vs.similarity_search(question, k=TOP_K)
        context     = _format_docs(docs)
        messages    = _build_messages(
            question, context, self._trim_history()
        )
        llm_sync = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            streaming=False,
        )
        response = llm_sync.invoke(messages)
        self._record_turn(question, response.content)
        return response.content, docs

    def summarize_repo(self) -> Iterator[str]:
        """Prints a friendly welcome message when the repository is first loaded."""
        yield "Hello! I am a naive RAG assistant for this repository. Ask me anything about the codebase."

    def clear_history(self) -> None:
        """Wipes the conversation history to start fresh."""
        self.history = []


# Helper function to easily load the Q&A engine from our main application

def load_qa(repo_id: str, repo_url: str = "") -> RepoQA:
    collection = f"{COLLECTION_NAME}_{repo_id}"
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return RepoQA(vs, repo_url=repo_url)
