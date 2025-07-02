import chromadb.api
import chromadb.api.client
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def load_chroma_vectorstore(collection_name: str, embedding, persist_directory='./db'):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    return vector_store