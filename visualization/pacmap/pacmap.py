import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pacmap

def show_pacmap(embeddings, labels=None, title="PaCMAP 2D Projection of Chunk Embeddings"):
    st.write("Running PaCMAP dimensionality reduction...")
    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    X_transformed = reducer.fit_transform(np.array(embeddings))

    fig, ax = plt.subplots()
    if labels is not None:
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='tab10', alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
        ax.add_artist(legend1)
    else:
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    st.pyplot(fig)