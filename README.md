# GitScope RAG - Codebase Chat Assistant

GitScope RAG is an assistant that helps you chat with and understand public GitHub code repositories. It reads files from a repository, splits them into readable parts, embeds them, and uses a smart context layout to answer your questions accurately.

---

## Weekly Progress and Daily Focus

This project was built over a 5-day cycle. Here is what we focused on each day:

| Day | Focus Area | One-Line Goal | Key Files |
| :--- | :--- | :--- | :--- |
| **Day 1** | Azure OpenAI Setup | Set up endpoints and basic streaming | [day-1/summary.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-1/summary.py) |
| **Day 2** | Chat Memory and Validation | Implement chat loops and support ticket verification | [day-2/pydantic_validation.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-2/pydantic_validation.py) |
| **Day 3** | Naive RAG Ingestion | Build and retrieve repo knowledge | [day-3/ingest.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-3/ingest.py), [day-3/main.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-3/main.py) |
| **Day 4** | Smart Context Windows | Organize knowledge into a clean context window | [day-4/qa_engine.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-4/qa_engine.py) |
| **Day 5** | Capstone Dashboard | Present the full assistant as a polished demo | [day-5/app.py](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/day-5/app.py) |

For a complete breakdown of each day's deliverables, see the [daily deliverables report](file:///d:/College/Intership/Bharatomni%20Technology/Development/week-1/week1_daily_deliverables.md).

---

## Tech Stack

- **Language:** Python 3.10+
- **LLM & Embeddings:** Azure OpenAI (gpt-4o-mini and text-embedding-ada-002)
- **Database:** ChromaDB (Local vector database)
- **Frameworks:** LangChain, Streamlit, GitPython, Pydantic

---

## How It Works

Here is the path a query takes in this system:

1. **User Question:** You ask a question about the repository.
2. **HyDE Rewrite:** The system writes a hypothetical code response first to use as a better search query.
3. **MMR Retrieval:** ChromaDB finds the top diverse chunks of code that match the query.
4. **Noise Filter:** Irrelevant parts like the raw directory tree or blank pages are filtered out.
5. **6-Layer Context Layout:** The prompt is assembled in order:
   - System Prompt (Rules)
   - Repository Card (Bird's-eye view)
   - Chat History (Last 6 turns)
   - Code Context (Selected code chunks)
   - Format Hint (Intent-based response formatting instructions)
   - User Question
6. **Response:** The LLM generates a streamed answer citing exact source files.

---

## Setup and Installation

### 1. Clone the project
```bash
git clone <repository-url>
cd week-1
```

### 2. Configure Environment Variables
Create a file named `.env` in the `week-1` root directory and add your Azure OpenAI details:
```env
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
```

### 3. Install Requirements
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run the Code

### Day 3 CLI Terminal Chat
To run the terminal-based Q&A loop:
```bash
cd day-3
python main.py
```
Type in a public GitHub URL and start chatting directly in your terminal.

### Day 5 Capstone Dashboard
To run the web interface:
```bash
cd day-5
streamlit run app.py
```
Open your browser to the link displayed (usually `http://localhost:8501`). Enter the GitHub URL in the sidebar to index it, and start asking questions.
