import streamlit as st

st.set_page_config(page_title="ML Model Evaluation Platform", layout="centered")

# ---------------- Welcome Page ----------------
st.title("👋 Welcome to the ML Model Evaluation Platform")

st.markdown(
    """
    Please log in to continue.  
    _(Static login for demo purposes — credentials are not validated)_
    """
)

# Login form
with st.form("login_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit:
    if name and email and password:
        st.success(f"✅ Welcome, {name}!")
        st.session_state["user_name"] = name
        st.session_state["user_email"] = email
        st.switch_page("pages/1_Model_Upload.py")  # go to Step 1
    else:
        st.error("⚠️ Please fill in all fields before proceeding.")
