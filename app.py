
import streamlit as st

st.set_page_config(
    page_title="Protein-Ligand Affinity Prediction",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Protein-Ligand Binding Affinity Prediction")

st.markdown("""
### GNN + 3D-CNN Fusion Deep Learning Model

This project predicts protein-ligand binding affinity
using two deep learning architectures.
""")

st.divider()

st.subheader("Model Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### GNN")
    st.write("Graph Neural Network")
    st.write("Extracts molecular and structural features.")

with col2:
    st.markdown("### 3D-CNN")
    st.write("3D Convolutional Neural Network")
    st.write("Learns spatial features from molecular voxels.")

with col3:
    st.markdown("### Fusion")
    st.write("Combined Model")
    st.write("Combines GNN and 3D-CNN representations.")

st.divider()

st.subheader("Upload Molecular Structures")

protein = st.file_uploader(
    "Upload Protein Pocket (PDB)",
    type=["pdb"]
)

ligand = st.file_uploader(
    "Upload Ligand Structure (SDF)",
    type=["sdf"]
)

if protein and ligand:
    st.success("Both files uploaded successfully.")

st.info(
    "Prediction functionality will be connected "
    "after training and saving the model."
)

st.divider()

st.caption("Project: Protein-Ligand Binding Affinity Prediction")
st.caption("Dataset: PDBbind")
