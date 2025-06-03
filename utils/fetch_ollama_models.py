import requests

def fetch_ollama_llm_models(ollama_api_url='http://localhost:11434'):
    try:
        response = requests.get(f"{ollama_api_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        llm_models = [model["name"] for model in models if "embed" not in model["name"].lower()]
        embedding_models = [model["name"] for model in models if "embed" in model["name"].lower()]
        return llm_models, embedding_models
    except requests.RequestException as e:
        print(f"Error fetching LLM models : {e}")
        return []