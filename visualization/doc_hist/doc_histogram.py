from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")

def show_hist(loaded_doc):
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(loaded_doc)]
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()