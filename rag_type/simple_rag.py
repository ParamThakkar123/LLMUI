from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from vector_databases.chroma import load_chroma_vectorstore
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import os
import uuid

def perform_simple_rag(
        llm, 
        embedding, 
        data, 
        query, 
        splittertype: str, 
        chunk_size: int, 
        chunk_overlap: int,
    ):
    # Use a unique persist directory for each run
    persist_directory = f'./db/{uuid.uuid4()}'
    os.makedirs(persist_directory, exist_ok=True)

    if splittertype == "Character":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splittertype == "Recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Invalid splitter type selected.")

    documents = []
    if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'page_content'):
        for doc in data:
            texts = text_splitter.split_text(doc.page_content)
            for t in texts:
                # Preserve original metadata if present, else add a default
                meta = dict(doc.metadata) if hasattr(doc, "metadata") else {}
                if "source" not in meta:
                    meta["source"] = "unknown"
                documents.append(Document(page_content=t, metadata=meta))
    elif isinstance(data, str):
        texts = text_splitter.split_text(data)
        for t in texts:
            documents.append(Document(page_content=t, metadata={"source": "user_input"}))
    else:
        raise ValueError("Unsupported data format for splitting.")

    vector_store = load_chroma_vectorstore(
        collection_name="simple_rag_collection",
        embedding=embedding,
        persist_directory=persist_directory
    )
    vector_store.add_documents(documents)
    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result