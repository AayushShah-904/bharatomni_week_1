"""
app.py — Streamlit UI for the GitHub RAG app.

Run with:
    streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GitHub Repo Explorer",
    page_icon="",
    layout="wide",
)

st.title("GitHub Repo Explorer")
st.caption("Paste a public GitHub repo URL and ask anything about the codebase.")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "repo_url"   not in st.session_state: st.session_state.repo_url   = ""
if "repo_id"    not in st.session_state: st.session_state.repo_id    = None
if "qa"         not in st.session_state: st.session_state.qa         = None
if "messages"   not in st.session_state: st.session_state.messages   = []
if "summarized" not in st.session_state: st.session_state.summarized = False


# ---------------------------------------------------------------------------
# Sidebar — repo input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Repository")

    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
        value=st.session_state.repo_url,
    )

    force = st.checkbox("Force re-ingest", value=False,
                        help="Re-clone and re-embed even if already indexed")

    ingest_btn = st.button("Load repo", type="primary", use_container_width=True)


    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summarized = False
        if st.session_state.qa:
            st.session_state.qa.clear_history()
        st.rerun()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

if ingest_btn and repo_url:
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.error("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set. Add them to your .env file.")
        st.stop()

    st.session_state.repo_url   = repo_url
    st.session_state.messages   = []
    st.session_state.summarized = False
    st.session_state.qa         = None

    with st.spinner("Cloning and indexing repo — this takes ~1–3 min for a medium repo ..."):
        try:
            from ingest import ingest
            vs, repo_id = ingest(repo_url, force_reingest=force)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.stop()

    with st.spinner("Building repo card and context (one-time, ~15 sec) ..."):
        try:
            from qa_engine import RepoQA
            st.session_state.qa      = RepoQA(vs, repo_url=repo_url)
            st.session_state.repo_id = repo_id
        except Exception as e:
            st.error(f"Context build failed: {e}")
            st.stop()

    st.success("Repo indexed! Ask anything below.")
    st.rerun()


# ---------------------------------------------------------------------------
# Auto-summary after first load
# ---------------------------------------------------------------------------

if st.session_state.qa and not st.session_state.summarized:
    st.session_state.summarized = True
    with st.chat_message("assistant"):
        st.write_stream(st.session_state.qa.summarize_repo())
    st.session_state.messages.append({
        "role": "assistant",
        "content": "(Project summary generated above)"
    })


# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask anything about the repo ...", disabled=st.session_state.qa is None):
    if st.session_state.qa is None:
        st.warning("Load a repo first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = st.write_stream(st.session_state.qa.stream_answer(prompt))

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------------
# Empty state prompt
# ---------------------------------------------------------------------------

if not st.session_state.qa:
    st.info("Enter a GitHub repository URL in the sidebar and click **Load repo** to get started.")