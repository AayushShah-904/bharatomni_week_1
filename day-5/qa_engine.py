# This file runs our advanced Q&A engine. It uses "context engineering" to structure the AI's inputs,
# rewrites the user query with HyDE for better retrieval, fetches relevant chunks from Chroma using MMR,
# and formats the output cleanly for the user.

from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# Basic configurations for searching and history-keeping

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "github_repo"
TOP_K           = 8    # chunks returned to LLM
FETCH_K         = 24   # MMR candidate pool (3x TOP_K)
MAX_HISTORY     = 6    # conversation turn pairs kept in context


# Rules that teach the AI how to behave and formatting expectations

SYSTEM_PROMPT = """\
You are an expert software engineer assistant. Your job is to help users deeply \
understand a GitHub codebase.

Rules:
- Cite every file you reference using backtick paths, e.g. `src/auth/jwt.py`.
- When showing code, quote only the relevant lines from the provided context.
- If the context is insufficient to answer, say so clearly — never hallucinate code.
- For architecture questions, explain how components relate based on actual imports \
and call patterns you can see in the code.
- Use markdown. Keep answers focused — no padding.
"""


# Automatically detects the user's intent to suggest the best answer format

def _format_hint(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["explain", "how does", "how do", "what does", "why"]):
        return (
            "Structure your answer as: "
            "(1) plain-English explanation, "
            "(2) the key code snippet with file path, "
            "(3) which other files are involved."
        )
    if any(w in q for w in ["where", "which file", "find", "locate", "what file"]):
        return (
            "Lead with the exact file path(s), then a one-sentence explanation of each. "
            "Be concise."
        )
    if any(w in q for w in ["summarize", "overview", "architecture", "structure", "design"]):
        return (
            "Answer with a bullet-point summary grouped by layer or responsibility. "
            "Include file paths for each point."
        )
    if any(w in q for w in ["list", "what are", "show all", "enumerate"]):
        return "Answer as a numbered or bulleted list. Include file paths."
    if any(w in q for w in ["diff", "change", "break", "impact", "affect", "depend"]):
        return (
            "Answer by tracing the dependency chain: start from the changed item "
            "and list what calls or imports it, with file paths."
        )
    return "Answer concisely with file path citations. Use markdown."


# Helper functions to format documents and package context for the LLM

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
    repo_card: str,
    history: list[dict],
    format_hint: str,
) -> list[dict]:
    """
    Assembles the 6-layer context window recipe in order:
    1. System prompt (role and rules)
    2. Repository card (high-level overview pinned at top)
    3. Conversation history (sliding window of Q&As)
    4. Retrieved code chunks (relevance-filtered source code)
    5. Output format hint (intent-based formatting expectations)
    6. User question (latest request)
    """
    # LAYER 1: System instructions
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # LAYER 2: Pinned repo card (high-level overview)
    if repo_card:
        messages.append({
            "role": "user",
            "content": f"## Repository overview\n\n{repo_card}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the repository overview. Ask me anything.",
        })

    # LAYER 3: Slide-windowed conversation history
    messages.extend(history)

    # LAYER 4, 5, and 6: Code Context, Format Hints, and the current Question
    messages.append({
        "role": "user",
        "content": (
            f"## Relevant code from the repository\n\n{context}\n\n" # LAYER 4: Retrieved code
            f"---\n\n"
            f"**Output format:** {format_hint}\n\n"                   # LAYER 5: Output format hint
            f"**Question:** {question}"                              # LAYER 6: User question
        ),
    })

    return messages


# Prompts and tools to create a quick 150-word overview of the repo on first load

REPO_CARD_PROMPT = """\
Using the repository structure and README excerpts below, write a repo card \
in under 150 words covering exactly these four points:

1. What this project does (1-2 sentences)
2. Tech stack (comma-separated list)
3. Key entry points (file paths where execution starts)
4. Top-level folder purposes (one line each)

Be factual and concise. Use only what you can see in the provided context.

---

{context}
"""

def build_repo_card(vs: Chroma, llm: AzureChatOpenAI) -> str:
    tree_docs   = vs.similarity_search("repository structure directory tree", k=1)
    readme_docs = vs.similarity_search("README project overview purpose install", k=3)
    context = _format_docs(tree_docs + readme_docs)

    response = llm.invoke([
        {"role": "system", "content": "You summarize software repositories concisely and accurately."},
        {"role": "user",   "content": REPO_CARD_PROMPT.format(context=context)},
    ])
    return response.content


# HyDE (Hypothetical Document Embeddings) writes a fake answer first to find better search results

HYDE_PROMPT = """\
A developer is asking this question about a codebase:

"{question}"

Write a short hypothetical code snippet or technical explanation (3-6 sentences) \
that would directly answer this question. Be specific — include realistic function \
names, variable names, and file references. This will be used as a search query, \
not shown to the user.
"""

def _hyde_retrieve(question: str, vs: Chroma, llm: AzureChatOpenAI) -> list[Document]:
    try:
        hypothetical = llm.invoke([
            {"role": "user", "content": HYDE_PROMPT.format(question=question)},
        ])
        search_text = hypothetical.content
    except Exception:
        search_text = question  # graceful fallback

    raw_docs = vs.max_marginal_relevance_search(search_text, k=TOP_K, fetch_k=FETCH_K)
    
    # Filter out noisy or redundant documents (e.g. __directory_tree__, empty/whitespace chunks)
    filtered_docs = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "")
        # The directory tree is already summarized in the repo card; we exclude it here to avoid noise
        if source == "__directory_tree__" or not doc.page_content.strip():
            continue
        filtered_docs.append(doc)
    return filtered_docs


# Main Q&A engine class

class RepoQA:
    """
    Our main engine that handles searching the repository, remembering chat context,
    and generating clean answers for the user.
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

        # [2] Built once, reused every call
        self.repo_card: str = build_repo_card(vectorstore, self.llm)

        # Holds a simple history of the conversation to handle follow-up questions
        self.history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        return self.history[-(MAX_HISTORY * 2):]

    def _record_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

    def stream_answer(self, question: str) -> Iterator[str]:
        """Finds relevant code, streams the answer in real-time, and adds sources."""
        docs        = _hyde_retrieve(question, self.vs, self.llm)
        context     = _format_docs(docs)
        format_hint = _format_hint(question)
        messages    = _build_messages(
            question, context, self.repo_card, self._trim_history(), format_hint
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
        """Finds relevant code and returns the complete answer and source documents at once."""
        docs        = _hyde_retrieve(question, self.vs, self.llm)
        context     = _format_docs(docs)
        format_hint = _format_hint(question)
        messages    = _build_messages(
            question, context, self.repo_card, self._trim_history(), format_hint
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
        """Streams a helpful detailed summary of the repository for onboarding."""
        expand_prompt = (
            f"Based on this repo card:\n\n{self.repo_card}\n\n"
            f"Write a fuller onboarding summary for a new developer in this format:\n\n"
            f"**What this project does** (2-3 sentences)\n\n"
            f"**Tech stack** (bullet list)\n\n"
            f"**Key entry points** (file paths + one-line description each)\n\n"
            f"**Main modules / folders** (brief description of each top-level area)\n\n"
            f"**How to run it** (if you can infer from the repo card)\n\n"
            f"Keep it concise. Use only information from the repo card."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": expand_prompt},
        ]
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history = []


# Helper function to easily initialize this class from our Streamlit app

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
