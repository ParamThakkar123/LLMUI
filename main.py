import streamlit as st
from utils.fetch_hf_models import fetch_huggingface_models, fetch_huggingface_embedding_models
from data_load.load_pdf import load_pdf
from rag_type.simple_rag import perform_simple_rag
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from tools.web_search import web_search_tool
from tools.web_crawl_tool import web_crawl_tool
from rag_type.agentic_rag import create_agent, run_agent
from utils.fetch_ollama_models import fetch_ollama_llm_models, fetch_ollama_embedding_models
from providers.ollama import load_ollama_model
from evals.evals import batch_bleu, bert_score, exact_match, f1, rouge_l, precision_at_k, recall_at_k
from visualization.kg_vis.knowledge_graph import build_kg
from visualization.doc_hist.doc_histogram import show_hist
from visualization.pacmap.pacmap import show_pacmap
import asyncio
from benchmarking.dataset_benchmarking import dataset_benchmarking
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        "RAG System Benchmarking",
        "Model Quantization",
        "Multi agent use case",
        "Dataset Benchmarking",
        "Visualization"
    )
)

# --- Retrieval Augmented Generation ---
if task_option == "Retrieval Augmented Generation":
    model_option = st.sidebar.selectbox(
        "Select a model provider",
        ("Huggingface", "Ollama"),
        key="rag_model_provider"
    )

    hf_model_name = None
    ollama_model_name = None
    embedding_model_name = None
    ollama_embedding_model = None

    if model_option == "Huggingface":
        if "hf_model_names" not in st.session_state:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
        model_names = st.session_state.get("hf_model_names", [])
        hf_model_name = st.sidebar.selectbox(
            "Select a Huggingface model",
            model_names,
            key="rag_hf_model"
        ) if model_names else None

        if "hf_embedding_models" not in st.session_state:
            with st.spinner("Fetching Huggingface embedding models..."):
                st.session_state.hf_embedding_models = fetch_huggingface_embedding_models()
        embedding_models = st.session_state.get("hf_embedding_models", [])
        embedding_model_name = st.sidebar.selectbox(
            "Select a Huggingface embedding model",
            embedding_models,
            key="rag_hf_embedding_model"
        ) if embedding_models else None

    elif model_option == "Ollama":
        if "ollama_model_names" not in st.session_state:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
        ollama_model_names = st.session_state.get("ollama_model_names", [])
        ollama_model_name = st.sidebar.selectbox(
            "Select an Ollama Model",
            ollama_model_names,
            key="rag_ollama_model"
        ) if ollama_model_names else None

        if "ollama_embedding_models" not in st.session_state:
            with st.spinner("Fetching Ollama embedding models..."):
                embedding_models = fetch_ollama_embedding_models()
                st.session_state.ollama_embedding_models = embedding_models
        ollama_embedding_models = st.session_state.get("ollama_embedding_models", [])
        ollama_embedding_model = st.sidebar.selectbox(
            "Select an Ollama embedding model",
            ollama_embedding_models,
            key="rag_ollama_embedding_model"
        ) if ollama_embedding_models else None

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
                "Character",
                "Recursive",
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

    pdf_data = None
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

            embedding_model = None
            if model_option == "Huggingface" and embedding_model_name:
                embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            elif model_option == "Ollama" and ollama_embedding_model:
                embedding_model = OllamaEmbeddings(model=ollama_embedding_model)

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

# --- RAG System Benchmarking ---
if task_option == "RAG System Benchmarking":
    st.sidebar.subheader("RAG System Benchmarking Configuration")

    if "rag_benchmark_models" not in st.session_state:
        st.session_state.rag_benchmark_models = []

    eval_model_provider = st.sidebar.selectbox(
        "Select Model Provider for Evaluation",
        ("Huggingface", "Ollama"),
        key="eval_model_provider"
    )

    llm_model_name = None
    embedding_model_name = None

    if eval_model_provider == "Huggingface":
        if "hf_model_names" not in st.session_state:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
        model_names = st.session_state.get("hf_model_names", [])
        llm_model_name = st.sidebar.selectbox(
            "Select a Huggingface model", model_names, key="eval_hf_llm_model"
        ) if model_names else None

        if "hf_embedding_models" not in st.session_state:
            with st.spinner("Fetching Huggingface embedding models..."):
                st.session_state.hf_embedding_models = fetch_huggingface_embedding_models()
        embedding_models = st.session_state.get("hf_embedding_models", [])
        embedding_model_name = st.sidebar.selectbox(
            "Select a Huggingface embedding model", embedding_models, key="eval_hf_embedding_model"
        ) if embedding_models else None

    elif eval_model_provider == "Ollama":
        if "ollama_model_names" not in st.session_state:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
        ollama_model_names = st.session_state.get("ollama_model_names", [])
        llm_model_name = st.sidebar.selectbox(
            "Select an Ollama model", ollama_model_names, key="eval_ollama_llm_model"
        ) if ollama_model_names else None

        if "ollama_embedding_models" not in st.session_state:
            with st.spinner("Fetching Ollama embedding models..."):
                embedding_models = fetch_ollama_embedding_models()
                st.session_state.ollama_embedding_models = embedding_models
        ollama_embedding_models = st.session_state.get("ollama_embedding_models", [])
        embedding_model_name = st.sidebar.selectbox(
            "Select an Ollama embedding model", ollama_embedding_models, key="eval_ollama_embedding_model"
        ) if ollama_embedding_models else None

    if st.sidebar.button("Add Model"):
        if llm_model_name and embedding_model_name:
            st.session_state.rag_benchmark_models.append({
                "provider": eval_model_provider,
                "llm_model": llm_model_name,
                "embedding_model": embedding_model_name
            })
            st.sidebar.success(f"Added {eval_model_provider} - {llm_model_name} / {embedding_model_name}")
        else:
            st.sidebar.warning("Please select both an LLM and an embedding model.")

    if st.session_state.rag_benchmark_models:
        st.sidebar.markdown("#### Models to Benchmark")
        for idx, m in enumerate(st.session_state.rag_benchmark_models):
            st.sidebar.write(f"{idx+1}. {m['provider']} - {m['llm_model']} / {m['embedding_model']}")

    # RAG and evaluation config
    vector_db_option = st.sidebar.selectbox(
        "Select the vector database you want to use",
        ("Qdrant", "Milvus", "Chroma", "FAISS"),
        key="eval_vector_db"
    )
    splitter_option = st.sidebar.selectbox(
        "Select Splitter Type",
        ("Character", "Recursive"),
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
        "Recall@k"
    ]
    selected_metrics = st.sidebar.multiselect(
        "Select Evaluation Metrics",
        available_metrics,
        default=["BLEU", "BERTScore"]
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

    if st.button("Evaluate All Models", key="eval_all_models_btn"):
        if not eval_query:
            st.warning("Please enter a question in the Evaluation Query box.")
        elif not ground_truths:
            st.warning("Please enter at least one ground truth.")
        elif not pdf_data:
            st.warning("Please upload a PDF file for context.")
        elif not st.session_state.rag_benchmark_models:
            st.warning("Please add at least one model to benchmark.")
        else:
            results = []
            for m in st.session_state.rag_benchmark_models:
                provider = m["provider"]
                llm_model_name = m["llm_model"]
                embedding_model_name = m["embedding_model"]

                # Load LLM
                chat_model = None
                if provider == "Huggingface":
                    llm_pipeline = HuggingFacePipeline.from_model_id(
                        model_id=llm_model_name,
                        task="text-generation",
                        pipeline_kwargs={"max_new_tokens": 100},
                        device=0 if torch.cuda.is_available() else -1,
                    )
                    chat_model = ChatHuggingFace(llm=llm_pipeline)
                elif provider == "Ollama":
                    chat_model = load_ollama_model(llm_model_name)

                embedding_model = None
                if provider == "Huggingface":
                    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
                elif provider == "Ollama":
                    embedding_model = OllamaEmbeddings(model=embedding_model_name)

                result = perform_simple_rag(
                    llm=chat_model,
                    embedding=embedding_model,
                    data=pdf_data,
                    query=eval_query,
                    splittertype=splitter_option,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                prediction = result["result"]
                row = {
                    "Provider": provider,
                    "LLM": llm_model_name,
                    "Embedding": embedding_model_name,
                    "Prediction": prediction
                }
                if "BLEU" in selected_metrics:
                    row["BLEU"] = batch_bleu([prediction], [ground_truths[0]])
                if "BERTScore" in selected_metrics:
                    bert = bert_score([prediction], [ground_truths[0]])
                    row["BERTScore_Precision"] = bert["BERTScore_precision"]
                    row["BERTScore_Recall"] = bert["BERTScore_Recall"]
                    row["BERTScore_F1"] = bert["BERTScore_F1"]
                if "Exact Match" in selected_metrics:
                    row["Exact Match"] = exact_match([prediction], [ground_truths[0]])
                if "F1" in selected_metrics:
                    row["F1"] = f1([prediction], [ground_truths[0]])
                if "ROUGE-L" in selected_metrics:
                    row["ROUGE-L"] = rouge_l([prediction], [ground_truths[0]])
                if "Precision@k" in selected_metrics:
                    row["Precision@k"] = precision_at_k([[prediction]], [ground_truths[0]])
                if "Recall@k" in selected_metrics:
                    row["Recall@k"] = recall_at_k([[prediction]], [ground_truths[0]])

                results.append(row)

            import pandas as pd
            st.markdown("### Benchmark Results")
            st.dataframe(pd.DataFrame(results))

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

# --- Fine Tuning ---
if task_option == "Fine Tuning":
    st.sidebar.subheader("Fine Tuning Dataset")
    fine_tune_dataset = st.sidebar.file_uploader(
        "Upload your fine tuning dataset (CSV, JSON, JSONL, TXT)",
        type=["csv", "json", "jsonl", "txt"]
    )
    if fine_tune_dataset is not None:
        st.write(f"Fine tuning dataset uploaded: **{fine_tune_dataset.name}**")

    fine_tuning_method = st.selectbox(
        "Select fine tuning method",
        [
            "LoRA",
            "Peft",
            "QLoRA",
            "GRPO",   
            "SFT",
            "RLHF"
        ]
    )

# --- Agentic AI ---
if task_option == "Agentic AI":
    st.sidebar.subheader("Agentic AI Configuration")

    model_option = st.sidebar.selectbox(
        "Select a model provider",
        ("Huggingface", "Ollama"),
        key="agentic_model_provider"
    )

    hf_model_name = None
    ollama_model_name = None

    if model_option == "Huggingface":
        if "hf_model_names" not in st.session_state:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
        model_names = st.session_state.get("hf_model_names", [])
        hf_model_name = st.sidebar.selectbox(
            "Select a Huggingface model",
            model_names,
            key="agentic_hf_model"
        ) if model_names else None

    elif model_option == "Ollama":
        if "ollama_model_names" not in st.session_state:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
        ollama_model_names = st.session_state.get("ollama_model_names", [])
        ollama_model_name = st.sidebar.selectbox(
            "Select an Ollama Model",
            ollama_model_names,
            key="agentic_ollama_model"
        ) if ollama_model_names else None

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

    available_tools = ["Web Search", "Web Crawl Tool"]
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
            elif selected_tool.lower() == "web crawl tool":
                st.session_state.agent_tools.append(web_crawl_tool)
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
        print(tools)
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
            print(agent)
            st.success("Agent created successfully!")

    st.markdown("---")
    agent_query = st.text_input("Ask your AI Agent a question:", key="agent_query_input")
    if st.button("Send to Agent"):
        agent = st.session_state.get("agent", None)
        if agent is not None:
            response = run_agent(agent, agent_query)
            st.write(response)
            st.write(response['messages'][-1].content)
        else:
            st.warning("Please create an agent first using the sidebar options.")

if task_option == "Dataset Benchmarking":
    model_option = st.sidebar.selectbox(
        "Select a model provider",
        ("Huggingface", "Ollama"),
        key="agentic_model_provider"
    )

    hf_model_name = None
    ollama_model_name = None

    if model_option == "Huggingface":
        if "hf_model_names" not in st.session_state:
            with st.spinner("Fetching Huggingface models..."):
                st.session_state.hf_model_names = fetch_huggingface_models()
        model_names = st.session_state.get("hf_model_names", [])
        hf_model_name = st.sidebar.selectbox(
            "Select a Huggingface model",
            model_names,
            key="agentic_hf_model"
        ) if model_names else None

    elif model_option == "Ollama":
        if "ollama_model_names" not in st.session_state:
            with st.spinner("Fetching Ollama models..."):
                llm_models, _ = fetch_ollama_llm_models()
                st.session_state.ollama_model_names = llm_models
        ollama_model_names = st.session_state.get("ollama_model_names", [])
        ollama_model_name = st.sidebar.selectbox(
            "Select an Ollama Model",
            ollama_model_names,
            key="agentic_ollama_model"
        ) if ollama_model_names else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Excel Dataset")
    excel_file = st.sidebar.file_uploader(
        "Upload your Excel file (must have 'Question' and 'Answer' columns)",
        type=["xlsx", "xls"],
        key="benchmark_excel_file"
    )

    # Add a button to load the model
    if st.sidebar.button("Load Model"):
        llm_model = None
        if model_option == "Huggingface" and hf_model_name:
            llm_pipeline = HuggingFacePipeline.from_model_id(
                model_id=hf_model_name,
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 100},
                device=0 if torch.cuda.is_available() else -1,
            )
            llm_model = ChatHuggingFace(llm=llm_pipeline)
        elif model_option == "Ollama" and ollama_model_name:
            llm_model = load_ollama_model(ollama_model_name)
        st.session_state.llm_model = llm_model
        st.success("Model loaded successfully!")

    # Only run benchmarking if both file and model are loaded
    llm_model = st.session_state.get("llm_model", None)
    if excel_file is not None and llm_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(excel_file.read())
            tmp_file_path = tmp_file.name
        st.success(f"Excel file uploaded: {excel_file.name}")
        dataset_benchmarking(tmp_file_path, llm_model)
    elif excel_file is not None and llm_model is None:
        st.warning("Please load a model before benchmarking.")