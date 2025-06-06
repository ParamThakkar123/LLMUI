from langchain_community.vectorstores import FAISS

def load_faiss_vectorstore(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db