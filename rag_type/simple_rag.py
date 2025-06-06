from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from vector_databases.chroma import load_chroma_vectorstore
from langchain.chains import RetrievalQA

def perform_simple_rag(
        llm, 
        embedding, 
        data, 
        query, 
        splittertype: str, 
        chunk_size: int, 
        chunk_overlap: int,
    ):
    if splittertype == "Character Text Splitter":
        text_splitter = CharacterTextSplitter(chunk_size, chunk_overlap)
    elif splittertype == "Recursive Character Text Splitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    else:
        raise ValueError("Invalid splitter type selected.")
    texts = text_splitter.split_text(data)

    db = load_chroma_vectorstore(texts, embeddings=embedding)
    retriever = db.as_retriever(search_kwargs={'k': 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    return result
