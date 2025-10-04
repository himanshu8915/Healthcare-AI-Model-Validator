import streamlit as st
from backend import chatbot

# ---------------- Page Config ----------------
st.set_page_config(page_title="Step 5: Chatbot Advisor", layout="wide")

st.markdown("### 💬 Step 5 of 5: Healthcare Model Advisor Chatbot")

st.info(
    "This chatbot acts as a **healthcare AI expert**. It uses your model’s evaluation metrics "
    "to suggest improvements, explain weaknesses, and provide healthcare-specific guidance."
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

# Chat input
if user_input := st.chat_input("Ask a question about your model..."):
    # Save user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    # Fetch metrics from session state (fall back if missing)
    metrics = st.session_state.get("metrics", {})
    fairness = st.session_state.get("fairness_metrics", {})
    operational = st.session_state.get("operational_metrics", {})

    # Get expert advice from backend
    with st.spinner("Thinking..."):
        response = chatbot.get_advice(metrics, fairness, operational)

    # Save + display response
    st.session_state.chat_history.append(("assistant", response))
    st.chat_message("assistant").markdown(response)

# Options
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Restart Chat"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("⬅️ Back to Evaluation"):
        st.switch_page("pages/3_Evaluation.py")
