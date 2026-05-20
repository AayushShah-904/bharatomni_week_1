import os
from enum import Enum
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class SupportTicket(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = Field(min_length=1)
    missing_information: List[str]
    suggested_next_step: str = Field(min_length=1)

parser = PydanticOutputParser(pydantic_object=SupportTicket)

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment="gpt-4.1-mini",
    api_version="2024-12-01-preview",
    temperature=0,
)

sample_transcript = """
I am having trouble logging into my account. I keep getting an error message that says
'Invalid credentials' even though I am sure I am entering the correct password.
Can you help me resolve this issue?
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support triage assistant. Return only structured output."),
    ("user", """Analyze the following customer message and format it exactly as instructed.

{format_instructions}

Customer message:
{message}
""")
])

chain = prompt | llm | parser

try:
    structured_llm = llm.with_structured_output(SupportTicket, method="json_schema")
    result = structured_llm.invoke(sample_transcript)
    print(result.model_dump_json(indent=2))
except ValidationError as e:
    print("Validation failed:")
    print(e)
except Exception as e:
    print("Other error:")
    print(e)
    