import streamlit as st
import os

st.title("Step 1: Upload AI Model")

if "model_uploaded" not in st.session_state:
    st.session_state.model_uploaded = False
if "model_path" not in st.session_state:
    st.session_state.model_path = None

model_file = st.file_uploader("Upload your AI model (.pkl or .onnx)", type=["pkl","onnx"])
if model_file:
    os.makedirs("temp_models", exist_ok=True)
    model_path = f"temp_models/{model_file.name}"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    st.session_state.model_uploaded = True
    st.session_state.model_path = model_path
    st.success(f"✅ Model uploaded: {model_file.name}")

if model_file is not None:
    st.success("✅ Model uploaded successfully!")
    if st.button("Next ➡️"):
        st.switch_page("pages/2_Test_Data.py")    
