"""
ingest.py — GitHub repo ingestion pipeline
Clones a repo, chunks code intelligently by language, embeds, and stores in Chroma.
"""

import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Generator

import git
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "github_repo"

# Extensions we care about, mapped to LangChain Language enum for smart splitting
LANGUAGE_MAP: dict[str, Language | None] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".ts":   Language.JS,       
    ".tsx":  Language.JS,
    ".jsx":  Language.JS,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".java": Language.JAVA,
    ".cpp":  Language.CPP,
    ".c":    Language.C,
    ".cs":   Language.CSHARP,
    ".rb":   Language.RUBY,
    ".md":   Language.MARKDOWN,
    ".mdx":  Language.MARKDOWN,
    ".html": Language.HTML,
    ".txt":  None,              # plain splitter
    ".env.example": None,
    ".yaml": None,
    ".yml":  None,
    ".json": None,
    ".toml": None,
}

# Files/dirs to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "vendor", "target",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "Cargo.lock",
    ".DS_Store", "thumbs.db",
}

MAX_FILE_BYTES = 150_000   # skip files larger than ~150 KB
CHUNK_SIZE     = 1_000     # tokens approx; LangChain uses chars internally
CHUNK_OVERLAP  = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clone_repo(url: str, target_dir: str) -> None:
    """Clone a GitHub repo (shallow) into target_dir."""
    print(f"  Cloning {url} ...")
    git.Repo.clone_from(url, target_dir, depth=1)


def repo_id_from_url(url: str) -> str:
    """Derive a stable short ID from the repo URL."""
    return hashlib.sha1(url.encode()).hexdigest()[:10]


def walk_repo(repo_dir: str) -> Generator[Path, None, None]:
    """Yield every relevant file path in the repo."""
    root = Path(repo_dir)
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # Skip unwanted directories
        parts = set(path.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() not in LANGUAGE_MAP and path.name not in LANGUAGE_MAP:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            print(f"    Skipping large file: {path.relative_to(root)}")
            continue
        yield path


def build_directory_tree(repo_dir: str, max_depth: int = 4) -> str:
    """
    Build a compact text tree of the repo structure.
    This is injected as a special 'meta' document so the LLM always has
    bird's-eye context about where things live.
    """
    root = Path(repo_dir)
    lines = [root.name + "/"]

    def _recurse(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, entry in enumerate(entries):
            if entry.name in SKIP_DIRS or entry.name.startswith("."):
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _recurse(entry, prefix + extension, depth + 1)

    _recurse(root, "", 1)
    return "\n".join(lines)


def load_and_chunk_file(path: Path, repo_dir: str) -> list[Document]:
    """Read a file and split it into Documents using the right splitter."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"    Could not read {path.name}: {e}")
        return []

    if not content.strip():
        return []

    rel_path = str(path.relative_to(repo_dir))
    ext = path.suffix.lower()
    lang = LANGUAGE_MAP.get(ext) or LANGUAGE_MAP.get(path.name)

    # Pick splitter
    if lang is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    metadata = {
        "source": rel_path,
        "file_name": path.name,
        "language": lang.value if lang else "text",
        "extension": ext,
    }

    chunks = splitter.create_documents([content], metadatas=[metadata])

    # Add chunk index so we can reconstruct order later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest(repo_url: str, force_reingest: bool = False) -> tuple[Chroma, str]:
    """
    Ingest a GitHub repo into Chroma.
    Returns (vectorstore, repo_id).
    Skips ingestion if already done (unless force_reingest=True).
    """
    repo_id = repo_id_from_url(repo_url)
    collection = f"{COLLECTION_NAME}_{repo_id}"

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    # Check if already ingested
    if not force_reingest and Path(CHROMA_DIR).exists():
        try:
            vs = Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
            if vs._collection.count() > 0:
                print(f"  Already ingested ({vs._collection.count()} chunks). Use force_reingest=True to redo.")
                return vs, repo_id
        except Exception:
            pass

    # Clone into a temp dir
    tmp = tempfile.mkdtemp()
    try:
        clone_repo(repo_url, tmp)

        # --- Layer 1: directory tree as a meta document ---
        print("  Building directory tree ...")
        tree_text = build_directory_tree(tmp)
        tree_doc = Document(
            page_content=f"# Repository structure\n\n```\n{tree_text}\n```",
            metadata={"source": "__directory_tree__", "file_name": "__tree__", "language": "text", "extension": ""},
        )

        # --- Layer 2 + 3: docs and source code ---
        print("  Walking and chunking files ...")
        all_docs: list[Document] = [tree_doc]
        file_count = 0

        for file_path in walk_repo(tmp):
            chunks = load_and_chunk_file(file_path, tmp)
            if chunks:
                all_docs.extend(chunks)
                file_count += 1

        print(f"  Processed {file_count} files → {len(all_docs)} chunks")

        # --- Embed and store ---
        print("  Embedding and storing in Chroma ...")
        # Chroma has a batch limit; process in batches of 500
        BATCH = 500
        vs = None
        for i in range(0, len(all_docs), BATCH):
            batch = all_docs[i : i + BATCH]
            if vs is None:
                vs = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=collection,
                    persist_directory=CHROMA_DIR,
                )
            else:
                vs.add_documents(batch)
            print(f"    Stored {min(i + BATCH, len(all_docs))}/{len(all_docs)} chunks ...")

        print(f"  Done. {len(all_docs)} chunks in collection '{collection}'.")
        return vs, repo_id

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/tiangolo/fastapi"
    vs, rid = ingest(url)
    print(f"Vectorstore ready. Repo ID: {rid}")
