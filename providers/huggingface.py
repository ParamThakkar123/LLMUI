from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_huggingface_model(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    return tokenizer, model

def load_huggingface_embedding(model: str):
    model = HuggingFaceEmbeddings(model)
    return model