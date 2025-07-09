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
import streamlit as st
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
        response = llm_model.invoke(prompt) if hasattr(llm_model, "invoke") else llm_model(prompt)
        if isinstance(response, dict) and "result" in response:
            pred = response["result"]
        elif hasattr(response, "content"):
            pred = response.content
        else:
            pred = str(response)
        predictions.append(pred)
        
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
        st.markdown("### Evaluation Metrics Plot")
        metrics_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = [
            "BLEU", "BERTScore F1", "Exact Match", "F1", "ROUGE-L",
            "ROUGE-1", "ROUGE-2", "Levenshtein Distance", 
            "Jaccard Similarity", "Jaccard Accuracy", 
            "Levenshtein Accuracy", "Batch Bleu"
        ]
        metrics_df[metrics_to_plot].plot(kind='bar', ax=ax)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics for Each Sample")
        ax.legend(loc='upper right')
        st.pyplot(fig)