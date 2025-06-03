import streamlit as st
from providers.huggingface import load_huggingface_model
from utils.fetch_hf_models import fetch_huggingface_models, fetch_huggingface_embedding_models
from utils.fetch_ollama_models import fetch_ollama_llm_models

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

data_option = st.sidebar.selectbox(
    "Select the type of data",
    (
        "PDF files",
        "Videos",
        "Images",
        "CSV files",
        "Github Repository",
        "Unstructured CSV",
        "Code files"
    )
)

hf_model_name = None
custom_hf_model_name = None
show_custom_hf_input = False

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
        st.sidebar.markdown("---")
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

embedding_model_name = None
custom_embedding_model_name = None
show_custom_embedding_input = False

if task_option == "Retrieval Augmented Generation":
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

if task_option == "Fine Tuning":
    st.sidebar.markdown("---")
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