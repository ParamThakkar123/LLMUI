import streamlit as st
from providers.huggingface import load_huggingface_model, load_huggingface_embedding
from utils.fetch_hf_models import fetch_huggingface_models, fetch_huggingface_embedding_models
from data_load.load_csv import load_unstructured_csv
from data_load.load_pdf import load_pdf
from rag_type.simple_rag import perform_simple_rag
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from tools.web_search import web_search_tool
from rag_type.agentic_rag import create_agent, run_agent
from utils.fetch_ollama_models import fetch_ollama_llm_models, fetch_ollama_embedding_models
from providers.ollama import load_ollama_model
from evals.evals import bleu_score, batch_bleu, bert_score
from langgraph.checkpoint.memory import MemorySaver
from visualization.kg_vis.knowledge_graph import build_kg
from visualization.doc_hist.doc_histogram import show_hist
from visualization.pacmap.pacmap import show_pacmap
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("Retrieval Augmented Generation Project")

st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate or set options.")

# Define model provider once at the top
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

task_option = st.sidebar.selectbox(
    "Select a task to be performed on LLMs",
    (
        "Retrieval Augmented Generation",
        "Fine Tuning",
        "LLM inference optimization",
        "Agentic AI",
        "RAG System Evaluation",
        "Model Quantization",
        "Multi agent use case",
        "Visualization"
    )
)

st.sidebar.subheader("Load Models")
hf_model_name = None
custom_hf_model_name = None
show_custom_hf_input = False
ollama_model_name = None
ollama_embedding_model = None
loaded_hf_model = None
loaded_hf_tokenizer = None
loaded_embedding_model = None
loaded_ollama_model = None

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

elif model_option == "Ollama":
    if "ollama_model_names" not in st.session_state:
        with st.sidebar:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
    ollama_model_names = st.session_state.get("ollama_model_names", [])
    if ollama_model_names:
        ollama_model_name = st.sidebar.selectbox(
            "Select an Ollama Model",
            ollama_model_names
        )
    else:
        st.sidebar.warning("Could not fetch ollama models")

    if st.sidebar.button("Load Ollama Model"):
        if isinstance(ollama_model_name, str) and ollama_model_name.strip():
            with st.spinner(f"Loading Huggingface model: {ollama_model_name} ..."):
                loaded_ollama_model = load_ollama_model(ollama_model_name)
            st.sidebar.success(f"Model '{ollama_model_name}' loaded successfully!")
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
            "Telegram",
            "Markdown Files"
        )
    )

    rag_type_option = st.sidebar.selectbox(
        "Select the type of RAG to perform on the data",
        (
            "Simple RAG",
            "Graph RAG",
            "Advanced RAG"
        )
    )

    vector_db_option = st.sidebar.selectbox(
        "Select the vector database you want to use",
        (
            "Qdrant",
            "Milvus",
            "Chroma",
            "FAISS"
        )
    )

    if data_option == "PDF files":
        splitter_option = st.sidebar.selectbox(
            "Select Splitter Type",
            (
                "Character Text Splitter",
                "Recursive Character Text Splitter",
            )
        )

    chunk_size = st.sidebar.number_input(
        "Enter chunk size for RAG",
        min_value=1,
        max_value=10000,
        value=1000,
        step=1,
        key="chunk_size"
    )

    chunk_overlap = st.sidebar.number_input(
        "Enter chunk overlap for RAG",
        min_value=0,
        max_value=10000,
        value=200,
        step=1,
        key="chunk_overlap"
    )

    memory_enabled = st.sidebar.toggle("Memory", value=False)
    st.write(f"Memory Enabled: {memory_enabled}")

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
            
            if st.sidebar.button("Load Embedding Model"):
                if isinstance(embedding_model_name, str) and embedding_model_name.strip():
                    with st.spinner(f"Loading Huggingface embedding model: {embedding_model_name} ..."):
                        loaded_embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
                    st.sidebar.success(f"Embedding model '{embedding_model_name}' loaded successfully!")
                else:
                    st.sidebar.error("Please select or enter a valid Huggingface embedding model name before loading.")
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
            if st.sidebar.button("Load Embedding Model"):
                if isinstance(embedding_model_name, str) and embedding_model_name.strip():
                    with st.spinner(f"Loading Huggingface embedding model: {embedding_model_name} ..."):
                        loaded_embedding_model = load_huggingface_embedding(embedding_model_name)
                    st.sidebar.success(f"Embedding model '{embedding_model_name}' loaded successfully!")
                else:
                    st.sidebar.error("Please select or enter a valid Huggingface embedding model name before loading.")

    if model_option == "Ollama":
        if "ollama_embedding_models" not in st.session_state:
            with st.sidebar:
                with st.spinner("Fetching Ollama Embedding Models"):
                    embedding_models = fetch_ollama_embedding_models()
                    st.session_state.ollama_embedding_models = embedding_models
            
        ollama_embedding_models = st.session_state.get("ollama_embedding_models", [])
        if ollama_embedding_models:
            ollama_embedding_model = st.sidebar.selectbox(
                "Select an Ollama Embedding Model",
                ollama_embedding_models
            )
            if st.sidebar.button("Load Ollama Embedding Model"):
                if isinstance(ollama_embedding_model, str) and ollama_embedding_model.strip():
                    embedding_model = OllamaEmbeddings(model=ollama_embedding_model)
                    st.sidebar.success(f"Ollama embedding model '{ollama_embedding_model}' loaded successfully!")
                else:
                    st.sidebar.error("Please select a valid Ollama embedding model.")
        else:
            st.sidebar.info("No Ollama embedding models loaded.")

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
        if rag_type_option == "Simple RAG":
            chat_model = None
            if model_option == "Huggingface" and hf_model_name:
                llm_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=hf_model_name,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 100},
                    device=0 if torch.cuda.is_available() else -1,
                )
                chat_model = ChatHuggingFace(llm=llm_pipeline)
            elif model_option == "Ollama" and ollama_model_name:
                chat_model = load_ollama_model(ollama_model_name)
            
            # Always set embedding_model based on the selected provider and embedding_model_name
            embedding_model = None
            if model_option == "Huggingface" and embedding_model_name:
                embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            elif model_option == "Ollama" and ollama_embedding_model:
                embedding_model = OllamaEmbeddings(model=ollama_embedding_model)

            # Now you can safely use embedding_model in perform_simple_rag
            result = perform_simple_rag(
                llm=chat_model,
                embedding=embedding_model,
                data=pdf_data,
                query=user_query,
                splittertype=splitter_option,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.write(result["result"])

if task_option == "Fine Tuning":
    st.sidebar.subheader("Fine Tuning Dataset")
    fine_tune_dataset = st.sidebar.file_uploader(
        "Upload your fine tuning dataset (CSV, JSON, JSONL, TXT)",
        type=["csv", "json", "jsonl", "txt"]
    )
    if fine_tune_dataset is not None:
        st.write(f"Fine tuning dataset uploaded: **{fine_tune_dataset.name}**")

if task_option == "Agentic AI":
    st.sidebar.subheader("Agentic AI Configuration")

    agent_name = st.sidebar.text_input(
        "Enter the name of the agent",
        value="Agent"
    )

    agent_description = st.sidebar.text_area(
        "Enter a description for the agent",
        placeholder="This agent is designed to perform various tasks."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Add Tool")

    available_tools = ["Web Search"]
    selected_tool = st.sidebar.selectbox("Select Tool", available_tools, key="tool_select")
    tool_description = st.sidebar.text_area("Tool Description", key="tool_description")

    serper_api_key = None
    if selected_tool.lower() == "web search":
        serper_api_key = st.sidebar.text_input("SERPER API Key", type="password", key="serper_api_key")

    if "agent_tools" not in st.session_state:
        st.session_state.agent_tools = []

    if st.sidebar.button("Add Tool"):
        if selected_tool and tool_description:
            if selected_tool.lower() == "web search":
                if not serper_api_key:
                    st.sidebar.error("Please provide the SERPER API key for the web search tool.")
                else:
                    st.session_state.agent_tools.append(web_search_tool(serper_api_key=serper_api_key))
                    st.sidebar.success(f"Tool {selected_tool} Added")
            else:
                st.session_state.agent_tools.append({
                    "name": selected_tool,
                    "description": tool_description
                })
                st.sidebar.success(f"Tool {selected_tool} Added")
        else:
            st.sidebar.error("Please provide a tool and a tool description.")

    prompt = st.sidebar.text_input("Enter a prompt for the agent", placeholder="Enter a prompt for the agent")
    if st.sidebar.button("Create Agent"):
        if model_option.lower() == "huggingface":
            model_id = hf_model_name
        elif model_option.lower() == "ollama":
            model_id = ollama_model_name
        else:
            model_id = None
        model_provider = model_option.lower()
        tools = st.session_state.agent_tools
        if not model_id:
            st.sidebar.error("Please select a model for the agent.")
        elif not prompt:
            st.sidebar.error("Please enter a prompt for the agent.")
        else:
            agent = create_agent(
                model_id=model_id,
                model_provider=model_provider,
                tools=tools,
                prompt=prompt
            )
            st.session_state.agent = agent
            st.success("Agent created successfully!")

    st.markdown("---")
    agent_query = st.text_input("Ask your AI Agent a question:", key="agent_query_input")
    if st.button("Send to Agent"):
        agent = st.session_state.get("agent", None)
        if agent is not None:
            response = run_agent(agent, agent_query)
            st.write(response.content)
        else:
            st.warning("Please create an agent first using the sidebar options.")

if task_option == "RAG System Evaluation":
    st.sidebar.subheader("RAG System Evaluation Configuration")

    eval_model_provider = st.sidebar.selectbox(
        "Select Model Provider for Evaluation",
        ("Huggingface", "Ollama"),
        key="eval_model_provider"
    )

    llm_model_name = None
    embedding_model_name = None
    embedding_model = None

    # LLM selection
    if eval_model_provider == "Huggingface":
        if "hf_model_names" not in st.session_state:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
        model_names = st.session_state.get("hf_model_names", [])
        llm_model_name = st.sidebar.selectbox(
            "Select a Huggingface model", model_names, key="eval_hf_llm_model"
        ) if model_names else None
    elif eval_model_provider == "Ollama":
        if "ollama_model_names" not in st.session_state:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
        ollama_model_names = st.session_state.get("ollama_model_names", [])
        llm_model_name = st.sidebar.selectbox(
            "Select an Ollama model", ollama_model_names, key="eval_ollama_llm_model"
        ) if ollama_model_names else None

    # Embedding model selection
    if eval_model_provider == "Huggingface":
        if "hf_embedding_models" not in st.session_state:
            with st.spinner("Fetching Huggingface embedding models..."):
                st.session_state.hf_embedding_models = fetch_huggingface_embedding_models()
        embedding_models = st.session_state.get("hf_embedding_models", [])
        embedding_model_name = st.sidebar.selectbox(
            "Select a Huggingface embedding model", embedding_models, key="eval_hf_embedding_model"
        ) if embedding_models else None
        if embedding_model_name:
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    elif eval_model_provider == "Ollama":
        if "ollama_embedding_models" not in st.session_state:
            with st.spinner("Fetching Ollama embedding models..."):
                embedding_models = fetch_ollama_embedding_models()
                st.session_state.ollama_embedding_models = embedding_models
        ollama_embedding_models = st.session_state.get("ollama_embedding_models", [])
        embedding_model_name = st.sidebar.selectbox(
            "Select an Ollama embedding model", ollama_embedding_models, key="eval_ollama_embedding_model"
        ) if ollama_embedding_models else None
        if embedding_model_name:
            embedding_model = OllamaEmbeddings(model=embedding_model_name)

    vector_db_option = st.sidebar.selectbox(
        "Select the vector database you want to use",
        ("Qdrant", "Milvus", "Chroma", "FAISS"),
        key="eval_vector_db"
    )

    splitter_option = st.sidebar.selectbox(
        "Select Splitter Type",
        ("Character Text Splitter", "Recursive Character Text Splitter"),
        key="eval_splitter"
    )

    chunk_size = st.sidebar.number_input(
        "Enter chunk size for RAG",
        min_value=1,
        max_value=10000,
        value=1000,
        step=1,
        key="eval_chunk_size"
    )
    chunk_overlap = st.sidebar.number_input(
        "Enter chunk overlap for RAG",
        min_value=0,
        max_value=10000,
        value=200,
        step=1,
        key="eval_chunk_overlap"
    )

    available_metrics = [
        "BLEU",
        "BERTScore",
        "Exact Match",
        "F1",
        "ROUGE-L",
        "Precision@k",
        "Context Precision@k"
    ]
    selected_metrics = st.sidebar.selectbox(
        "Select Evaluation Metric",
        available_metrics,
        key="eval_metric"
    )

    st.markdown("### Enter your question")
    eval_query = st.text_input("Evaluation Query", key="eval_query")

    st.markdown("### Add Ground Truths (one per line)")
    ground_truths_input = st.text_area("Ground Truths", placeholder="Enter each ground truth on a new line", key="eval_ground_truths")
    ground_truths = [gt.strip() for gt in ground_truths_input.split('\n') if gt.strip()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload PDF File")
    pdf_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"], key="eval_pdf_file")
    pdf_data = None
    if pdf_file is not None:
        st.write(f"PDF file upload: **{pdf_file.name}**")
        with st.spinner("Loading PDF files..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            pdf_data = load_pdf(tmp_file_path)
        st.success("Successfully loaded PDF data")
        st.write(pdf_data)

    if st.button("Evaluate", key="eval_evaluate_btn"):
        if not eval_query:
            st.warning("Please enter a question in the Evaluation Query box.")
        elif not ground_truths:
            st.warning("Please enter at least one ground truth.")
        elif not pdf_data:
            st.warning("Please upload a PDF file for context.")
        elif not llm_model_name or not embedding_model_name:
            st.warning("Please select both a model and an embedding model.")
        else:
            chat_model = None
            if eval_model_provider == "Huggingface" and llm_model_name:
                llm_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=llm_model_name,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 100},
                    device=0 if torch.cuda.is_available() else -1,
                )
                chat_model = ChatHuggingFace(llm=llm_pipeline)
            elif eval_model_provider == "Ollama" and llm_model_name:
                chat_model = load_ollama_model(llm_model_name)

            embedding_model = None
            if eval_model_provider == "Huggingface" and embedding_model_name:
                embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            elif eval_model_provider == "Ollama" and embedding_model_name:
                embedding_model = OllamaEmbeddings(model=embedding_model_name)

            # Now you can safely use embedding_model in perform_simple_rag
            result = perform_simple_rag(
                llm=chat_model,
                embedding=embedding_model,
                data=pdf_data,
                query=eval_query,
                splittertype=splitter_option,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            prediction = result.result

            if selected_metrics == "BLEU":
                score = batch_bleu([prediction], [ground_truths[0]])
                st.success(f"BLEU Score: {score:.4f}")
            elif selected_metrics == "BERTScore":
                score = bert_score([prediction], [ground_truths[0]])
                st.success(f"BERTScore Precision: {score['BERTScore_precision']:.4f}")
                st.success(f"BERTScore Recall: {score['BERTScore_Recall']:.4f}")
                st.success(f"BERTScore F1: {score['BERTScore_F1']:.4f}")

            st.markdown("### Model Prediction")
            st.write(prediction)
            st.markdown("### Ground Truth(s)")
            for gt in ground_truths:
                st.write(gt)

if task_option == "Visualization":
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ("Knowledge Graph", "Histogram", "Clustering", "PacMap"),
        key="visualization_type"
    )

    if visualization_type == "Histogram":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload PDF File")
        pdf_file = st.sidebar.file_uploader(
            "Upload your PDF file for histogram",
            type=["pdf"],
            key="histogram_pdf_file"
        )
        if pdf_file is not None:
            st.write(f"PDF file uploaded: **{pdf_file.name}**")
            with st.spinner("Loading PDF file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                from data_load.load_pdf import load_pdf
                loaded_doc = load_pdf(tmp_file_path)
            st.success("Successfully loaded PDF data")
            st.write("Generating histogram...")
            show_hist(loaded_doc)

    if visualization_type == "Knowledge Graph":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload PDF File")
        pdf_file = st.sidebar.file_uploader(
            "Upload your PDF file for Knowledge Graph",
            type=["pdf"],
            key="kg_pdf_file"
        )

        llm_model_name = None
        if model_option == "Huggingface":
            if "hf_model_names" not in st.session_state:
                with st.spinner("Fetching Huggingface models... "):
                    st.session_state.hf_model_names = fetch_huggingface_models()
            model_names = st.session_state.get("hf_model_names", [])
            llm_model_name = st.sidebar.selectbox(
                "Select a Huggingface model for KG", model_names, key="kg_hf_llm_model"
            ) if model_names else None

        elif model_option == "Ollama":
            if "ollama_model_names" not in st.session_state:
                with st.spinner("Fetching Ollama models..."):
                    llm_models, _ = fetch_ollama_llm_models()
                    st.session_state.ollama_model_names = llm_models
            ollama_model_names = st.session_state.get("ollama_model_names", [])
            llm_model_name = st.sidebar.selectbox(
                "Select an Ollama model for KG", ollama_model_names, key="kg_ollama_llm_model"
            ) if ollama_model_names else None

        if pdf_file is not None and llm_model_name:
            st.write(f"PDF file uploaded : **{pdf_file.name}**")
            with st.spinner("Loading PDF file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                loaded_doc = load_pdf(tmp_file_path)
            st.success("Successfully loaded PDF data")
            st.write("Building and displaying knowledge graph...")

            if model_option == "Huggingface":
                llm_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=llm_model_name,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 100},
                    device=0 if torch.cuda.is_available() else -1,
                )
                chat_model = ChatHuggingFace(llm=llm_pipeline)
            elif model_option == "Ollama":
                chat_model = load_ollama_model(llm_model_name)

            asyncio.run(build_kg(chat_model, loaded_doc))

    if visualization_type == "PacMap":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload PDF File")
        pdf_file = st.sidebar.file_uploader(
            "Upload your PDF file for PaCMAP",
            type=["pdf"],
            key="pacmap_pdf_file"
        )
        if pdf_file is not None:
            st.write(f"PDF file uploaded: **{pdf_file.name}**")
            with st.spinner("Loading PDF file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                from data_load.load_pdf import load_pdf
                loaded_doc = load_pdf(tmp_file_path)
            st.success("Successfully loaded PDF data")
            st.write("Generating chunk embeddings and projecting with PaCMAP...")

            embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = [doc.page_content for doc in loaded_doc]
            chunks = text_splitter.create_documents(texts)
            chunk_texts = [doc.page_content for doc in chunks]
            embeddings = embedding_model.embed_documents(chunk_texts)
            show_pacmap(embeddings, title="PaCMAP 2D Projection of PDF Chunk Embeddings")