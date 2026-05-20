"""
qa_engine.py — Context-engineered Q&A over an ingested GitHub repo.

Context window structure (every call):
  [1] System prompt         — role, rules, citation format
  [2] Repo card             — stable 150-word summary generated once after ingest
  [3] Conversation history  — last N turn pairs, stored slim (no injected context)
  [4] Retrieved chunks      — top-k via HyDE query rewriting + MMR de-duplication
  [5] Output format hint    — tailored per detected question type
  [6] User question
Directory tree = map-the structure

Docs = guidebook-the purpose
 
Code chunks = detailed pages-the exact code

Dependency graph = lines showing how everything is connected-the relationships between code pieces
rewrites the user query with HyDE for better retrieval,fetches relevant chunks from Chroma using MMR
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
FETCH_K         = 24   # MMR candidate pool (3x TOP_K)
MAX_HISTORY     = 6    # conversation turn pairs kept in context


# ---------------------------------------------------------------------------
# [1] System prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# [5] Output format hints — detected from question intent
# ---------------------------------------------------------------------------

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
    repo_card: str,
    history: list[dict],
    format_hint: str,
) -> list[dict]:
    """
    Assemble the full context-engineered message list:
      system -> repo card -> history -> current question + chunks + format hint
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # [2] Repo card as a pinned exchange at the top
    if repo_card:
        messages.append({
            "role": "user",
            "content": f"## Repository overview\n\n{repo_card}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the repository overview. Ask me anything.",
        })

    # [3] Windowed conversation history
    messages.extend(history)

    # [4] + [5] + [6] Chunks + format hint + question
    messages.append({
        "role": "user",
        "content": (
            f"## Relevant code from the repository\n\n{context}\n\n"
            f"---\n\n"
            f"**Output format:** {format_hint}\n\n"
            f"**Question:** {question}"
        ),
    })

    return messages


# ---------------------------------------------------------------------------
# [2] Repo card builder — called once after ingest
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# [4] HyDE retrieval
# ---------------------------------------------------------------------------

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

    return vs.max_marginal_relevance_search(search_text, k=TOP_K, fetch_k=FETCH_K)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RepoQA:
    """
    Context-engineered Q&A over a Chroma vectorstore.

    Context window per call:
      [1] System prompt
      [2] Repo card (stable, generated once after ingest)
      [3] Windowed conversation history (last MAX_HISTORY turn pairs)
      [4] HyDE-retrieved + MMR-deduplicated chunks
      [5] Output format hint (per question type)
      [6] User question
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

        # [3] Slim history: list of {role, content} dicts (bare Q+A only, no injected context)
        self.history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        return self.history[-(MAX_HISTORY * 2):]

    def _record_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

    def stream_answer(self, question: str) -> Iterator[str]:
        """Stream answer tokens. Appends source citations, then records the turn."""
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
        """Non-streaming answer. Returns (text, source_docs)."""
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
        """Stream a fuller onboarding summary built from the repo card."""
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
