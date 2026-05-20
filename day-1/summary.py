import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://bo-ai-dev-api.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

sample_transcript = """
John: Welcome everyone. Today we need to clear up the Q3 product roadmap.
Sarah: The frontend team is blocked on the API authentication endpoints. We need those by Friday, or the login page will be delayed.
Mike: I can assign Dave to wrap up those auth endpoints. He should have them ready by Thursday afternoon.
John: Perfect. Moving on to marketing—Sarah, did we lock in the budget for the Paris campaign?
Sarah: Not yet, I am still waiting for finance approval. I'll ping them tomorrow morning.
John: Great. Let's meet again next Monday.
"""

summary_prompt = f"""Analyze the following meeting transcript. Provide a 2-sentence summary, followed by a bulleted list of Action Items 
(include who is responsible if mentioned):\n\n{sample_transcript}"""
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an expert executive assistant. You analyze text accurately and answer follow-up questions based strictly on the provided context.",
        },
        {
            "role": "user",
            "content": summary_prompt,
        }
    ],
    stream=True,
    # max_completion_tokens=13107,
    # temperature=1.0,
    # top_p=1.0,
    # frequency_penalty=0.0,
    # presence_penalty=0.0,
    model=deployment
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

client.close()
 


# Weak prompt
# Summarize this document.
# Improved prompt
# You are a project assistant. Summarize the document for a project lead in 5 bullet points. Focus on goals, deadlines, risks, and decisions.
# Do not include minor details. Add an Open Questions section for anything important that is unclear.


# Weak prompt
# Find the action items.
# Improved prompt
# You are a meeting notes assistant. Extract all action items from the notes below. Return a table with the columns: Owner, Task, Deadline, 
# and Status. If the owner or deadline is missing, write Not specified. Include only tasks that are clearly assigned or strongly implied.


# Weak prompt
# Classify this feedback.
# Improved prompt
# You are a customer support analyst. Classify each feedback item into one category: billing, technical, account, or other. Return 
# JSON with the fields id, category, and reason. If a message could fit multiple categories, choose the most important one and explain the decision briefly.
