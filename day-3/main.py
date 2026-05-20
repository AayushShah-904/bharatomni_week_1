"""
main.py — Terminal entry point for Day 3 Naive RAG.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import ingest
from qa_engine import RepoQA

def main():
    load_dotenv()
    
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set in .env file.")
        sys.exit(1)
        

    print("Welcome to Day 3 Naive RAG - GitHub Repo Explorer (CLI)")
    
    repo_url = input("Enter a public GitHub repository URL:\n> ").strip()
    if not repo_url:
        print("Invalid URL. Exiting.")
        sys.exit(1)
        
    force_input = input("Force re-ingest? (y/N): ").strip().lower()
    force_reingest = force_input in ["y", "yes"]
    
    print("\nCloning and indexing repository. This might take a moment...")
    try:
        vs, repo_id = ingest(repo_url, force_reingest=force_reingest)
    except Exception as e:
        print(f"\nIngestion failed: {e}")
        sys.exit(1)
        
    print("\nInitializing Q&A engine...")
    try:
        qa = RepoQA(vs, repo_url=repo_url)
    except Exception as e:
        print(f"\nEngine initialization failed: {e}")
        sys.exit(1)
        
    print("\n" + "-" * 60)
    # Stream the onboarding/greeting message
    for chunk in qa.summarize_repo():
        print(chunk, end="", flush=True)
    print("\n" + "-" * 60)
    
    print("\nCommands:")
    print("  Type 'exit', 'quit', or 'q' to end the session.")
    print("  Type 'clear' or 'reset' to clear the conversation history.")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break
            
        if not prompt:
            continue
            
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break
            
        if prompt.lower() in ["clear", "reset"]:
            qa.clear_history()
            print("Conversation history cleared.")
            continue
            
        print("\nAssistant: ", end="", flush=True)
        try:
            for chunk in qa.stream_answer(prompt):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nAn error occurred while generating the answer: {e}")

if __name__ == "__main__":
    main()
