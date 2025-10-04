import streamlit as st
from backend import data_generation
import os
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(page_title="Step 2: Test Data", layout="wide")

st.markdown("### 🧪 Step 2 of 5: Upload or Generate Test Data")

# Initialize session states
if "test_data_ready" not in st.session_state:
    st.session_state.test_data_ready = False
if "test_data_path" not in st.session_state:
    st.session_state.test_data_path = None

# ---------------- Option Selection ----------------
st.info("Choose whether you already have a test dataset or want to generate synthetic data.")

test_data_option = st.radio(
    "Do you have a test dataset?",
    ["📂 Yes, I have a CSV", "⚡ No, generate synthetic data"],
    horizontal=True
)

# ---------------- Upload CSV Option ----------------
if test_data_option.startswith("📂"):
    test_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])
    if test_file:
        os.makedirs("temp_data", exist_ok=True)
        test_data_path = f"temp_data/{test_file.name}"
        df = pd.read_csv(test_file)

        # Rename target column if needed
        if "Churn" in df.columns:
            df = df.rename(columns={"Churn": "label"})

        df.to_csv(test_data_path, index=False)
        st.session_state.test_data_ready = True
        st.session_state.test_data_path = test_data_path

        st.success(f"✅ Test data uploaded: {test_file.name}")

        # Preview uploaded data
        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df.head(10), use_container_width=True)

# ---------------- Synthetic Data Option ----------------
else:
    st.write("Provide details to generate synthetic test data:")
    with st.form("synthetic_data_form"):
        model_desc = st.text_area("🧾 Describe your model", placeholder="e.g., A churn prediction model for telecom users")
        features_desc = st.text_area("📊 Describe the features / parameters", placeholder="e.g., tenure, monthly charges, data usage, etc.")
        num_samples = st.number_input("🔢 Number of synthetic samples", min_value=10, max_value=10000, value=100)
        
        generate_btn = st.form_submit_button("🚀 Generate Synthetic Test Data")

    if generate_btn:
        test_data_path = data_generation.generate_synthetic_csv(model_desc, features_desc, num_samples)
        st.session_state.test_data_ready = True
        st.session_state.test_data_path = test_data_path
        st.success(f"✅ Synthetic test data generated: {test_data_path}")

        # Preview generated data
        if test_data_path and os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            with st.expander("🔍 Preview Generated Data"):
                st.dataframe(df.head(10), use_container_width=True)

# ---------------- Next Step Navigation ----------------
if st.session_state.test_data_ready:
    st.markdown("---")
    st.success("🎉 Test data is ready for evaluation!")
    if st.button("➡️ Proceed to Evaluation Metrics"):
        st.switch_page("pages/3_Evaluation.py")
