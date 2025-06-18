import chromadb.api
import chromadb.api.client
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def load_chroma_vectorstore(collection_name: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    chromadb.api.client.AdminClient.clear_system_cache()
    ef = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = Chroma(collection_name=collection_name, embedding_function=ef, persist_directory='./db')
    return vector_store