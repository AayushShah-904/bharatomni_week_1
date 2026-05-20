import os
from enum import Enum
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Define valid categories for incoming support tickets using standard python Enums
class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"

# Define the level of urgency for the tickets
class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

# Use Pydantic to describe the structure we want the AI to output.
# Pydantic will enforce that the category/urgency match our enums, and strings are not empty.
class SupportTicket(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = Field(min_length=1)
    missing_information: List[str]
    suggested_next_step: str = Field(min_length=1)

# Set up LangChain's Pydantic output parser using our model schema
parser = PydanticOutputParser(pydantic_object=SupportTicket)

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize our Azure ChatOpenAI connection
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment="gpt-4.1-mini",
    api_version="2024-12-01-preview",
    temperature=0,
)

# A sample customer support request message we want the model to analyze
sample_transcript = """
I am having trouble logging into my account. I keep getting an error message that says
'Invalid credentials' even though I am sure I am entering the correct password.
Can you help me resolve this issue?
"""

# Draft a template that tells the system its role and shows how the user wants the data formatted
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support triage assistant. Return only structured output."),
    ("user", """Analyze the following customer message and format it exactly as instructed.

{format_instructions}

Customer message:
{message}
""")
])

# Create a chain that flows the prompt into the LLM, then parses the raw output to match our schema
chain = prompt | llm | parser

# Call the model asking for a structured JSON response matching the SupportTicket schema
try:
    structured_llm = llm.with_structured_output(SupportTicket, method="json_schema")
    result = structured_llm.invoke(sample_transcript)
    print(result.model_dump_json(indent=2))
except ValidationError as e:
    # Print out nicely if the JSON structure doesn't match our schema rules
    print("Validation failed:")
    print(e)
except Exception as e:
    # Print out any other network or API errors
    print("Other error:")
    print(e)

    