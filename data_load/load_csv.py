from langchain_community.document_loaders.csv_loader import CSVLoader, UnstructuredCSVLoader

def load_csv(file_path: str):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def load_unstructured_csv(file_path: str):
    loader = UnstructuredCSVLoader(file_path=file_path)
    data = loader.load()
    return data