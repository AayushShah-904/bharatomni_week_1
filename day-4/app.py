from ingest import ingest
from qa_engine import RepoQA

def main():
    repo_url = input("Enter GitHub repo URL: ").strip()
    vs, repo_id = ingest(repo_url)
    qa = RepoQA(vs, repo_url=repo_url)

    print("\nRepo loaded. Type questions. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit","q"):
            break

        print("Assistant: ", end="", flush=True)
        for chunk in qa.stream_answer(question):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()