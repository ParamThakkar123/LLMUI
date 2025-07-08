import pandas as pd
from data_load.load_excel import load_excel
from evals.evals import bleu_score, bert_score, exact_match, f1, rouge_l
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

    if not {"Question", "Answer"}.issubset(df.columns):
        st.error("Excel file must contain 'Question' and 'Answer' columns.")
        return

    questions = df["Question"].tolist()
    ground_truths = df["Answer"].tolist()
    predictions = []

    prompt_template = "You are an expert assistant. Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"

    results = []
    table_placeholder = st.empty()

    for q, gt in zip(questions, ground_truths):
        prompt = prompt_template.format(question=q)
        response = llm_model.invoke(prompt) if hasattr(llm_model, "invoke") else llm_model(prompt)
        if isinstance(response, dict) and "result" in response:
            pred = response["result"]
        elif hasattr(response, "content"):
            pred = response.content
        else:
            pred = str(response)
        predictions.append(pred)

        bleu = bleu_score(pred, gt)
        bert = bert_score([pred], [gt])
        em = exact_match([pred], [gt])
        f1_score = f1([pred], [gt])
        rouge = rouge_l([pred], [gt])
        results.append({
            "Question": q,
            "Ground Truth": gt,
            "Prediction": pred,
            "BLEU": bleu,
            "BERTScore_F1": bert["BERTScore_F1"],
            "Exact Match": em,
            "F1": f1_score,
            "ROUGE-L": rouge
        })

        results_df = pd.DataFrame(results)
        table_placeholder.dataframe(results_df)

    if results:
        st.markdown("### Evaluation Metrics Plot")
        metrics_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ["BLEU", "BERTScore_F1", "Exact Match", "F1", "ROUGE-L"]
        metrics_df[metrics_to_plot].plot(kind='bar', ax=ax)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics for Each Sample")
        ax.legend(loc='upper right')
        st.pyplot(fig)