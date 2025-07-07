from pydantic import BaseModel
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import StructuredTool

class SearchInput(BaseModel):
    query: str

def web_search_tool(serper_api_key: str) -> StructuredTool:
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    def answer(query: str) -> str:
        return search.run(query)
    
    tool = StructuredTool.from_function(
        name="Web Search Tool",
        description="Searches the web for any queries.",
        func=answer,
        args_schema=SearchInput
    )

    return tool
