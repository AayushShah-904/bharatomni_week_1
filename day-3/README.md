# Day 3: Retrieval-Augmented Generation

This directory implements the core RAG pipeline (load, chunk, embed, store, retrieve, answer) for a software repository.

## Components

- **`ingest.py`**: Handles cloning, filtering, chunking, embedding, and storing the repository contents in Chroma.
- **`qa_engine.py`**: The Q&A assistant class (`RepoQA`) that retrieves code chunks using standard similarity search and streams the grounded answer.
- **`main.py`**: Terminal entry point providing a CLI chat loop interface.
- **`rag.py`**: Local policy documents RAG demonstration.

## How to Run

1. Make sure your `.env` file contains your Azure OpenAI credentials.
2. Run the CLI application:
   ```bash
   python main.py
   ```
3. Enter the public GitHub repository URL when prompted.
