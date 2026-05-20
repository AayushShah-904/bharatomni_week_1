"""
qa_engine.py — Naive RAG Q&A over an ingested GitHub repo.
"""

from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "github_repo"
TOP_K           = 8    # chunks returned to LLM
MAX_HISTORY     = 6    # conversation turn pairs kept in context


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer assistant. Your job is to help users deeply \
understand a GitHub codebase. Answer the user's question based on the provided code context. \
If the context is insufficient to answer, say so clearly — never hallucinate code. \
Use markdown. Keep answers focused.
"""


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        path = doc.metadata.get("source", "unknown")
        lang = doc.metadata.get("language", "")
        parts.append(f"### `{path}`\n```{lang}\n{doc.page_content}\n```")
    return "\n\n".join(parts)


def _build_messages(
    question: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    """
    Assemble the message list:
      system -> history -> current question + chunks
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RepoQA:
    """
    Naive Q&A over a Chroma vectorstore.

    Context window per call:
      [1] System prompt
      [2] Windowed conversation history (last MAX_HISTORY turn pairs)
      [3] Retrieved chunks (via similarity search)
      [4] User question
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

        # Slim history: list of {role, content} dicts (bare Q+A only, no injected context)
        self.history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        return self.history[-(MAX_HISTORY * 2):]

    def _record_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

    def stream_answer(self, question: str) -> Iterator[str]:
        """Stream answer tokens. Appends source citations, then records the turn."""
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
        """Non-streaming answer. Returns (text, source_docs)."""
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
        """Stream a simple greeting/onboarding message."""
        yield "Hello! I am a naive RAG assistant for this repository. Ask me anything about the codebase."

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history = []


# ---------------------------------------------------------------------------
# Convenience loader (used by app.py)
# ---------------------------------------------------------------------------

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
