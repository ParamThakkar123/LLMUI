from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def load_huggingface_model(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    return tokenizer, model

def load_huggingface_embedding(model: str):
    model = SentenceTransformer(model)
    return model