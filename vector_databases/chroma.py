from langchain_chroma import Chroma

def load_chroma_vectorstore(docs, embeddings):
    db = Chroma.from_documents(docs, embeddings)
    return db