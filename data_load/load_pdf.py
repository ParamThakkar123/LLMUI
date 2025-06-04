from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file: str):
    loader = PyPDFLoader(file_path=file)
    data = loader.load()
    return data