from langchain_ollama import ChatOllama

def load_ollama_model(model:str):
    llm = ChatOllama(model=model)
    return llm