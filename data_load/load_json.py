from langchain_community.document_loaders import JSONLoader

def load_json_file(file_path: str):
    loader = JSONLoader(file_path=file_path)
    data = loader.load()
    return data