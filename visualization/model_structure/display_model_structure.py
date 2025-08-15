import streamlit as st
def display_model_structure(model):
    st.markdown("### Model Structure")
    with st.expander("Show Model Summary"):
        st.code(str(model), language="python")

        # Top-level children
        top_children = dict(model.named_children())
        selected_top = st.selectbox("Select top-level module", list(top_children.keys()), key="top_level_module")
        if selected_top:
            st.write(f"**{selected_top}**")
            st.code(str(top_children[selected_top]), language="python")

            # If layers exist, show layer selection
            if hasattr(top_children[selected_top], "layers"):
                layers = getattr(top_children[selected_top], "layers")
                if hasattr(layers, "__len__"):
                    layer_indices = list(range(len(layers)))
                    selected_layer_idx = st.selectbox("Select layer", layer_indices, key="layer_idx")
                    layer_obj = layers[selected_layer_idx]
                    st.write(f"**Layer {selected_layer_idx}**")
                    st.code(str(layer_obj), language="python")

                    # If attention exists in layer
                    if hasattr(layer_obj, "self_attn"):
                        st.write("**Attention Module**")
                        st.code(str(layer_obj.self_attn), language="python")
                    if hasattr(layer_obj, "mlp"):
                        st.write("**MLP Module**")
                        st.code(str(layer_obj.mlp), language="python")
                    if hasattr(layer_obj, "input_layernorm"):
                        st.write("**Input LayerNorm**")
                        st.code(str(layer_obj.input_layernorm), language="python")
                    if hasattr(layer_obj, "post_attention_layernorm"):
                        st.write("**Post Attention LayerNorm**")
                        st.code(str(layer_obj.post_attention_layernorm), language="python")