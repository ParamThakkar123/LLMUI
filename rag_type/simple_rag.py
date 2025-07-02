from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from vector_databases.chroma import load_chroma_vectorstore
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import shutil
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

    if splittertype == "Character Text Splitter":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splittertype == "Recursive Character Text Splitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Invalid splitter type selected.")

    if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'page_content'):
        text = "\n".join(doc.page_content for doc in data)
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("Unsupported data format for splitting.")

    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]

    vector_store = load_chroma_vectorstore(
        collection_name="simple_rag_collection",
        embedding=embedding,
        persist_directory=persist_directory
    )
    vector_store.add_documents(documents)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result