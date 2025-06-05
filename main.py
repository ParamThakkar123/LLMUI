import streamlit as st
from providers.huggingface import load_huggingface_model, load_huggingface_embedding
from utils.fetch_hf_models import fetch_huggingface_models, fetch_huggingface_embedding_models
from data_load.load_csv import load_unstructured_csv
from data_load.load_pdf import load_pdf
import tempfile

st.title("Retrieval Augmented Generation Project")

st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate or set options.")

task_option = st.sidebar.selectbox(
    "Select a task to be performed on LLMs",
    (
        "Retrieval Augmented Generation",
        "Fine Tuning",
        "LLM inference optimization",
        "Agentic AI",
        "Multi agent use case"
    )
)

st.sidebar.subheader("Load Models")
model_option = st.sidebar.selectbox(
    "Select a model provider",
    (
        "Huggingface", 
        "Mistral", 
        "OpenAI", 
        "Deepseek",
        "Ollama",
        "Azure",
        "Nvidia",
        "xAI",
        "Anthropic",
        "OpenRouter",
        "Novita AI"
    )
)

hf_model_name = None
custom_hf_model_name = None
show_custom_hf_input = False

loaded_hf_model = None
loaded_hf_tokenizer = None
if model_option == "Huggingface":
    if "hf_model_names" not in st.session_state:
        with st.sidebar:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
    model_names = st.session_state.get("hf_model_names", [])
    if model_names:
        hf_model_name = st.sidebar.selectbox(
            "Select a Huggingface model",
            model_names
        )
        if "show_custom_hf_input" not in st.session_state:
            st.session_state.show_custom_hf_input = False
        if st.sidebar.button("Can't find my model"):
            st.session_state.show_custom_hf_input = not st.session_state.show_custom_hf_input
        if st.session_state.show_custom_hf_input:
            custom_hf_model_name = st.sidebar.text_input(
                "Enter your Huggingface model name:"
            )
            if custom_hf_model_name:
                hf_model_name = custom_hf_model_name
    else:
        st.sidebar.warning("Could not fetch Huggingface models.")
        if "show_custom_hf_input" not in st.session_state:
            st.session_state.show_custom_hf_input = True
        if st.session_state.show_custom_hf_input:
            custom_hf_model_name = st.sidebar.text_input(
                "Enter Huggingface model name manually:"
            )
            if custom_hf_model_name:
                hf_model_name = custom_hf_model_name

    if st.sidebar.button("Load Model"):
        if isinstance(hf_model_name, str) and hf_model_name.strip():
            with st.spinner(f"Loading Huggingface model: {hf_model_name} ..."):
                loaded_hf_tokenizer, loaded_hf_model = load_huggingface_model(hf_model_name)
            st.sidebar.success(f"Model '{hf_model_name}' loaded successfully!")
        else:
            st.sidebar.error("Please select or enter a valid Huggingface model name before loading.")

embedding_model_name = None
custom_embedding_model_name = None
show_custom_embedding_input = False

if task_option == "Retrieval Augmented Generation":

    data_option = st.sidebar.selectbox(
        "Select the type of data",
        (
            "PDF files",
            "Videos",
            "Youtube Video",
            "Images",
            "CSV files",
            "Github Repository",
            "Unstructured CSV",
            "JSON files",
            "Code files",
            "Tweets",
            "Reddit",
            "Google Drive",
            "Whatsapp Messages",
            "Telegram"
        )
    )

    rag_type_option = st.sidebar.selectbox(
        "Select the type of RAG to perform on the data",
        (
            "Simple RAG",
            "Graph RAG",
        )
    )

    if model_option == "Huggingface":
        if "hf_embedding_models" not in st.session_state:
            with st.sidebar:
                with st.spinner("Fetching Huggingface Embedding models.."):
                    embed_models = fetch_huggingface_embedding_models()
                    st.session_state.hf_embedding_models = embed_models
        embedding_models = st.session_state.get("hf_embedding_models", [])
        if embedding_models:
            embedding_model_name = st.sidebar.selectbox(
                "Select a Huggingaface embedding model",
                embedding_models
            )
            if "show_custom_embedding_input" not in st.session_state:
                st.session_state.show_custom_embedding_input = False
            if st.sidebar.button("Can't find my embedding model"):
                st.session_state.show_custom_embedding_input = not st.session_state.show_custom_embedding_input
            if st.session_state.show_custom_embedding_input:
                custom_embedding_model_name = st.sidebar.text_input(
                    "Enter your Huggingface embedding model name:"
                )
                if custom_embedding_model_name:
                    embedding_model_name = custom_embedding_model_name
        else:
            st.sidebar.warning("Could not fetch Huggingface embedding models.")
            if "show_custom_embedding_input" not in st.session_state:
                st.session_state.show_custom_embedding_input = True
            if st.session_state.show_custom_embedding_input:
                custom_embedding_model_name = st.sidebar.text_input(
                    "Enter Huggingface embedding model name manually:"
                )
                if custom_embedding_model_name:
                    embedding_model_name = custom_embedding_model_name

    if data_option == "PDF files":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload PDF File")
        pdf_file = st.sidebar.file_uploader(
            "Upload your PDF file",
            type=["pdf"]
        )
        if pdf_file is not None:
            st.write(f"PDF file upload: **{pdf_file.name}**")
            with st.spinner("Loading PDF files..."):
                # Save uploaded file to a temporary file and pass its path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                pdf_data = load_pdf(tmp_file_path)
            st.success("Successfully loaded PDF data")
            st.write(pdf_data)

    if data_option == "Images":
        st.sidebar.markdown("---")
        st.sidebar.header("Upload Image File")
        image_file = st.sidebar.file_uploader(
            "Upload your image file",
            type=["png", "jpg", "jpeg", "bmp", "gif"]
        )
        if image_file is not None:
            st.write(f"Image file uploaded: **{image_file.name}**")

    if data_option == "CSV files" or data_option == "Unstructured CSV":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload CSV file")
        csv_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=["csv"]
        )
        if csv_file is not None:
            st.write(f"CSV file uploaded: **{csv_file.name}**")

    if data_option == "Unstructured CSV":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload CSV file")
        csv_file = st.sidebar.file_uploader(
            "Upload your Unstructured CSV file",
            type=["csv"]
        )
        if csv_file is not None:
            st.write(f"CSV file uploaded: **{csv_file.name}**")
            with st.spinner("Loading Unstructured CSV file..."):
                csv_data = load_unstructured_csv(csv_file)
            st.success("Unstructured CSV file loaded successfully.")
            st.write(csv_data)

    if data_option == "JSON files":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload JSON File")
        json_file = st.sidebar.file_uploader(
            "Upload your JSON file",
            type=["json"]
        )
        if json_file is not None:
            st.write(f"JSON file uploaded: **{json_file.name}**")

    user_query = st.text_input("Enter your text query:", key="rag_query")
    if user_query:
        st.write(f"Your query: {user_query}")

    memory_enabled = st.sidebar.toggle("Memory", value=False)
    st.write(f"Memory Enabled: {memory_enabled}")

if task_option == "Fine Tuning":
    st.sidebar.subheader("Fine Tuning Dataset")
    fine_tune_dataset = st.sidebar.file_uploader(
        "Upload your fine tuning dataset (CSV, JSON, JSONL, TXT)",
        type=["csv", "json", "jsonl", "txt"]
    )
    if fine_tune_dataset is not None:
        st.write(f"Fine tuning dataset uploaded: **{fine_tune_dataset.name}**")

st.write(f"Selected Model Provider: **{model_option}**")
if model_option == "Huggingface" and hf_model_name:
    st.write(f"Selected Huggingface Model: **{hf_model_name}**")
if task_option == "Retrieval Augmented Generation" and embedding_model_name:
    st.write(f"Selected Embedding Model: **{embedding_model_name}**")