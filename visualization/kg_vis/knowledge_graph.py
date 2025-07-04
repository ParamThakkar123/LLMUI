import streamlit as st
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from pyvis.network import Network
import tempfile

graph = Neo4jGraph(refresh_schema=False)

async def build_kg(llm, loaded_doc):
    # Check if KG already exists (nodes and relationships present)
    nodes = graph.query("MATCH (n) RETURN n LIMIT 1")
    rels = graph.query("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1")
    if nodes and rels:
        st.info("Knowledge graph already exists. Fetching and rendering...")
    else:
        # Build KG if not present
        if isinstance(loaded_doc, list) and hasattr(loaded_doc[0], "page_content"):
            documents = loaded_doc
        elif isinstance(loaded_doc, list) and isinstance(loaded_doc[0], str):
            documents = [Document(page_content=text) for text in loaded_doc]
        else:
            st.warning("Unsupported document format for KG.")
            return

        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
        graph.add_graph_documents(graph_documents)
        st.success("Knowledge graph built and stored in Neo4j!")

    # Visualization using pyvis
    net = Network(notebook=False, directed=True)
    nodes = graph.query("MATCH (n) RETURN n")
    rels = graph.query("MATCH (n)-[r]->(m) RETURN n, r, m")
    node_ids = set()
    for record in nodes:
        node = record['n']
        node_id = node.element_id if hasattr(node, "element_id") else str(node)
        node_label = node.get("name", node_id)
        net.add_node(node_id, label=node_label)
        node_ids.add(node_id)
    for record in rels:
        n = record['n']
        m = record['m']
        r = record['r']
        from_id = n.element_id if hasattr(n, "element_id") else str(n)
        to_id = m.element_id if hasattr(m, "element_id") else str(m)
        rel_label = r.type if hasattr(r, "type") else str(r)
        if from_id in node_ids and to_id in node_ids:
            net.add_edge(from_id, to_id, label=rel_label)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        st.components.v1.html(open(tmp_file.name, "r", encoding="utf-8").read(), height=600, scrolling=True)