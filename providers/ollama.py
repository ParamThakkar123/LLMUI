from langchain_ollama import ChatOllama
import requests

def load_ollama_model(model: str, ollama_api_url='http://localhost:11434'):
    # Check if the model exists locally
    try:
        response = requests.get(f"{ollama_api_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        if model not in model_names:
            # Pull the model if not present
            pull_resp = requests.post(f"{ollama_api_url}/api/pull", json={"name": model})
            pull_resp.raise_for_status()
    except Exception as e:
        print(f"Error checking or pulling Ollama model: {e}")

    llm = ChatOllama(model=model)
    return llm