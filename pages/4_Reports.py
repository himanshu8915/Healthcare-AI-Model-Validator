import streamlit as st
from backend import evaluation

st.title("Step 4: Download Reports")

if not st.session_state.get("metrics"):
    st.warning("Run evaluation first!")
else:
    json_data = {**st.session_state.metrics, **st.session_state.fairness, **st.session_state.operational_metrics}
    st.download_button("Download JSON Report", data=str(json_data), file_name="evaluation_report.json")
    
    pdf_path = evaluation.generate_report(
        st.session_state.metrics,
        st.session_state.fairness,
        st.session_state.operational_metrics
    )
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF Report", data=f, file_name="evaluation_report.pdf")
col1, col2 = st.columns([1,1])
if col1.button("⬅️ Back"):
    st.switch_page("pages/3_Evaluation.py")
if col2.button("Next ➡️"):
    st.switch_page("pages/5_Chatbot.py")