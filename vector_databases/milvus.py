from langchain_community.vectorstores import Milvus

def load_milvus_vectorstore(docs, embeddings, host:str, port:str):
    db = Milvus.from_documents(docs, embeddings, connection_args={"host": host, "port": port})
    return db