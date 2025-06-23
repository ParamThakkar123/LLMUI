from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

def create_agent(model_id: str, model_provider: str, tools: list, prompt: str):
    llm = init_chat_model(model_id, model_provider=model_provider, temperature=0)
    agent = create_react_agent(llm, tools, prompt=prompt)
    return agent

def run_agent(agent, query: str):
    events = agent.stream(
        {
            "messages": [
                ("user", str(query))
            ]
        },
        stream_mode="values"
    )

    for event in events:
        print(event["messages"][-1])