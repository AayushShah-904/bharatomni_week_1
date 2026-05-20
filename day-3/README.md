# GitHub RAG — v1

Ask natural language questions about any public GitHub repository.

## Setup

```bash
# 1. Clone or copy this folder, then install dependencies
pip install -r requirements.txt

# 2. Add your Azure OpenAI credentials
cp .env.example .env
# Edit .env with your actual values:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_VERSION
#   AZURE_OPENAI_DEPLOYMENT_NAME          (chat model deployment)
#   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME (embeddings deployment)

# 3. Run the app
streamlit run app.py
```

## How it works

| Step | What happens |
|------|-------------|
| **Clone** | Shallow-clones the repo into a temp directory via `gitpython` |
| **Filter** | Skips binaries, lock files, node_modules, large files (>150 KB) |
| **Chunk** | Splits code at function/class boundaries using `RecursiveCharacterTextSplitter.from_language()` — not arbitrary character counts |
| **Embed** | `text-embedding-3-small` via OpenAI |
| **Store** | Local Chroma vector database (persisted to `./chroma_db/`) |
| **Retrieve** | MMR (Maximum Marginal Relevance) search — top 8 chunks, de-duplicated |
| **Answer** | GPT-4o with a code-aware system prompt and source citations |

## File structure

```
.
├── app.py          # Streamlit UI
├── ingest.py       # Clone → chunk → embed → store
├── qa_engine.py    # Retrieve → answer (streaming)
├── requirements.txt
└── .env.example
```

## Supported languages

Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, C#, Ruby,
Markdown, HTML, and plain text/config files (YAML, JSON, TOML).

## Upgrading to v2

- Replace Chroma with a hosted vector DB (Pinecone, Weaviate)
- Add BM25 hybrid search alongside vector search
- Use `tree-sitter` for multi-language AST-based chunking
- Add a LangGraph agent with multiple tools (file viewer, grep, graph traversal)
- Add GitHub webhook for incremental re-indexing on new commits
