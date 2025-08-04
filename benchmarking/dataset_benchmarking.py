import pandas as pd
from data_load.load_excel import load_excel
from evals.evals import (
    bleu_score, 
    bert_score, 
    exact_match, 
    f1, 
    rouge_l,
    rouge_1,
    rouge_2,
    batch_bleu, 
    levenshtein_distance,
    jaccard_similarity,
    levenshtein_similarity_accuracy,
    jaccard_accuracy,
    fertility_score
)
import torch
import streamlit as st
import time
import matplotlib.pyplot as plt

def dataset_benchmarking(excel_path, llm_model):
    data = load_excel(excel_path)
    if isinstance(data, list) and hasattr(data[0], "page_content"):
        df = pd.read_excel(excel_path)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_excel(excel_path)

    mcq_columns = {"Option A", "Option B", "Option C", "Option D"}
    is_mcq = mcq_columns.issubset(df.columns)

    if not {"Question", "Answer"}.issubset(df.columns):
        st.error("Excel file must contain 'Question' and 'Answer' columns.")
        return

    questions = df["Question"].tolist()
    ground_truths = df["Answer"].tolist()
    predictions = []

    results = []
    total_output_tokens = 0
    Tx = None
    Ty = None
    table_placeholder = st.empty()

    for idx, (q, gt) in enumerate(zip(questions, ground_truths)):
        if is_mcq:
            options = [
                f"A. {df.loc[idx, 'Option A']}",
                f"B. {df.loc[idx, 'Option B']}",
                f"C. {df.loc[idx, 'Option C']}", 
                f"D. {df.loc[idx, 'Option D']}"
            ]
            options_text = "\n".join(options)
            prompt = (
                "You are an expert assistant. Select the most correct option (A, B, C, or D) for the following multiple choice question. "
                "Provide only the option letter as the answer.\n\n"
                f"Question: {q}\n{options_text}\nAnswer:"
            )
        else:
            prompt = (
                "You are an expert assistant. The questions can be comprehensive or multiple choice. "
                "If the question is comprehensive give as much accurate answer as possible but if the question is a multiple choice select the option which you think is the most correct. "
                "Answer the following question concisely:\n\n"
                f"Question: {q}\nAnswer:"
            )

        e2e_start_time = time.time()
        if Tx is None:
            Tx = e2e_start_time
        start_time = time.time()
        first_token_time = None
        pred = ""
        if hasattr(llm_model, "stream"):
            for i, token in enumerate(llm_model.stream(prompt)):
                if i == 0:
                    first_token_time = time.time()
                if hasattr(token, "content"):
                    pred += str(token.content)
                else:
                    pred += str(token)
            if first_token_time is None:
                first_token_time = time.time()
        else:
            response = llm_model.invoke(prompt) if hasattr(llm_model, "invoke") else llm_model(prompt)
            if isinstance(response, dict) and "result" in response:
                pred = response["result"]
            elif hasattr(response, "content"):
                pred = response.content
            else:
                pred = str(response)
            first_token_time = time.time()
        
        e2e_end_time = time.time()
        Ty = e2e_end_time
        e2e_latency = e2e_end_time - e2e_start_time
        time_to_first_token = first_token_time - start_time
        predictions.append(pred)
        
        output_tokens = len(pred.split())
        total_output_tokens += output_tokens

        itl = 0
        if output_tokens > 1:
            itl = (e2e_latency - time_to_first_token) / (output_tokens - 1)

        tps_per_user = output_tokens / e2e_latency if e2e_latency > 0 else 0

        bat_bleu = batch_bleu([pred], [gt])
        bleu = bleu_score(pred, gt)
        bert = bert_score([pred], [gt])
        em = exact_match([pred], [gt])
        f1_score = f1([pred], [gt])
        lev_dist = levenshtein_distance([pred], [gt])
        jaccard = jaccard_similarity([pred], [gt])
        jaccard_acc = jaccard_accuracy([pred], [gt])
        lev_acc = levenshtein_similarity_accuracy([pred], [gt])
        fert_score = fertility_score([pred])
        roug_l = rouge_l([pred], [gt])
        roug_1 = rouge_1([pred], [gt])
        roug_2 = rouge_2([pred], [gt])
        results.append({
            "Question": q,
            "Ground Truth": gt,
            "Prediction": pred,
            "Time to First Token(s)": time_to_first_token,
            "E2E Latency (s)": e2e_latency,
            "Inter-token Latency": itl,
            "TPS per User": tps_per_user,
            "BLEU": bleu,
            "BERTScore F1": bert["BERTScore_F1"],
            "BERTScore precision": bert["BERTScore_precision"],
            "BERTScore Recall": bert["BERTScore_Recall"],
            "Exact Match": em,
            "F1": f1_score,
            "Batch Bleu": bat_bleu,
            "Levenshtein Distance": lev_dist,
            "Jaccard Similarity": jaccard,
            "Jaccard Accuracy": jaccard_acc,
            "Levenshtein Accuracy": lev_acc,
            "Fertility Score": fert_score,
            "ROUGE-1": roug_1,
            "ROUGE-2": roug_2,
            "ROUGE-L": roug_l
        })

        results_df = pd.DataFrame(results)
        table_placeholder.dataframe(results_df)

    if results:
        benchmark_duration = Ty - Tx if Ty and Tx and Ty > Tx else 1e-6
        total_tps = total_output_tokens / benchmark_duration
        rps = len(results) / benchmark_duration

        st.markdown("### Benchmark Summary")
        st.write(f"**Total Tokens Per Second (TPS)**: {total_tps:.2f}")
        st.write(f"**Requests Per Second (RPS)**: {rps:.4f}")

        st.markdown("### Evaluation Metrics Plot")
        metrics_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = [
            "BLEU", "BERTScore F1", "Exact Match", "F1", "ROUGE-L",
            "ROUGE-1", "ROUGE-2", "Levenshtein Distance", 
            "Jaccard Similarity", "Jaccard Accuracy", 
            "Levenshtein Accuracy", "Batch Bleu", 
            "TPS per User", "Inter-token Latency"
        ]
        valid_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        metrics_df.plot(kind='bar', ax=valid_metrics, ax=ax)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics for Sample")
        ax.legend(loc='upper right')
        st.pyplot(fig)

        if torch and torch.cuda.is_available():
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            gpu_mem_free = gpu_mem_total - gpu_mem_reserved

            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.bar(
                ["GPU Allocated (GB)", "GPU Reserved (GB)", "GPU Free (GB)"],
                [gpu_mem_alloc, gpu_mem_reserved, gpu_mem_free],
                color=["#2ca02c", "#d62728", "#9467bd"]
            )
            ax3.set_ylim(0, gpu_mem_total)
            ax3.set_ylabel("Memory (GB)")
            ax3.set_title("GPU Memory Usage")
            st.pyplot(fig3)
        else:
            st.info("No GPU detected or torch not installed. GPU usage will not be displayed.")