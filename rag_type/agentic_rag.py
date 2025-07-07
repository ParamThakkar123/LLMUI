from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

def create_agent(model_id: str, model_provider: str, tools: list, prompt: str):
    llm = init_chat_model(model_id, model_provider=model_provider, temperature=0)
    llm.bind_tools(tools)
    
    # Default system prompt if user-provided prompt is empty or insufficient
    default_prompt = """
    You are an intelligent assistant with access to the following tools: {tool_names}.
    For queries requiring real-time data, external information, or data beyond your knowledge cutoff, use the appropriate tool (e.g., 'Web Search Tool' for general queries or 'Web Crawl Tool' for detailed content from a specific website).
    If a query involves recent or specific information (e.g., stock prices, news), prioritize using the 'Web Search Tool' or 'Web Crawl Tool' to fetch accurate data.
    Always provide a clear and concise response based on the information available or retrieved.
    """
    
    # Format tool names for the prompt
    tool_names = ", ".join([tool.name for tool in tools]) if tools else "None"
    final_prompt = prompt if prompt.strip() else default_prompt.format(tool_names=tool_names)
    agent = create_react_agent(llm, tools, prompt=final_prompt)
    return agent

def run_agent(agent, query):
   response = agent.invoke({'messages': [HumanMessage(query)]})
  
   for message in response['messages']:
       print(
           f"{message.__class__.__name__}: {message.content}"
       )  # Print message class name and its content
      
       print("-" * 20, end="\n")
  
   return response