from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import os

os.environ["SERPER_API_KEY"] = "e5d1106caed5747b9774f708088544929d35117a"
llm = init_chat_model("llama3.2:1b", model_provider="ollama", temperature=0)
search = GoogleSerperAPIWrapper(serper_api_key="e5d1106caed5747b9774f708088544929d35117a")

# Define a Pydantic schema for the tool input
class SearchInput(BaseModel):
    query: str

def intermediate_answer(query: str) -> str:
    return search.run(query)

tools = [
    StructuredTool.from_function(
        name="Intermediate_Answer",
        description="useful for when you need to ask with search",
        func=intermediate_answer,
        args_schema=SearchInput
    )
]

agent = create_react_agent(llm, tools)
question = input("Ask a Question: ")
events = agent.stream(
    {
        "messages": [
            ("user", str(question))
        ]
    },
    stream_mode="values"
)

for event in events:
    print(event["messages"][-1])