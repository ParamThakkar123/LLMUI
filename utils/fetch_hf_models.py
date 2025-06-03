import requests

def fetch_huggingface_models():
    url = f"https://huggingface.co/api/models"
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json()
        return [model['id'] for model in models]
    else:
        return []
    
def fetch_huggingface_embedding_models():
    url = f"https://huggingface.co/api/models"
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json()
        embedding_models = [
            model['id'] for model in models
            if "embedding" in model['id'].lower() or "sentence-transformers" in model['id'].lower()
        ]
        return embedding_models
    else:
        return []