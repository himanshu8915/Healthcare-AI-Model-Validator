import streamlit as st
from backend import evaluation,chatbot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Step 3️⃣: Model Evaluation & Dashboard")

# --- Check prerequisites ---
if not st.session_state.get("model_uploaded") or not st.session_state.get("test_data_ready"):
    st.warning("⚠️ Please upload the model and test data first!")
else:
    # --- Run Evaluation ---
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            # Compute metrics
            st.session_state.metrics = evaluation.evaluate_model(
                st.session_state.model_path, st.session_state.test_data_path
            )

            # Compute fairness metrics if applicable
            df_test = pd.read_csv(st.session_state.test_data_path)
            sensitive_cols = [c for c in df_test.columns if df_test[c].nunique() <= 10 and c != "label"]
            if sensitive_cols:
                sensitive_col = st.selectbox("Sensitive column for fairness metrics", sensitive_cols)
                st.session_state.fairness = evaluation.compute_fairness_metrics(
                    st.session_state.model_path, st.session_state.test_data_path, sensitive_column=sensitive_col
                )
            else:
                st.session_state.fairness = {"Note": "No categorical columns available for fairness analysis."}

            # Operational metrics
            st.session_state.operational_metrics = evaluation.compute_operational_metrics(
                st.session_state.model_path, st.session_state.test_data_path
            )
            
            st.success("✅ Evaluation complete!")

    # --- Display Dashboard ---
    if st.session_state.get("metrics"):
        metrics = st.session_state.metrics
        fairness = st.session_state.fairness
        operational_metrics = st.session_state.operational_metrics
        model_path = st.session_state.model_path
        test_data_path = st.session_state.test_data_path

        # --- Tabs ---
        tabs = st.tabs(["Performance", "Fairness & Bias", "Calibration", "Visualizations", "Chatbot Advice"])

        # ---------------- Performance Metrics ----------------
        with tabs[0]:
            st.subheader("🏆 Basic Performance Metrics")
            cols = st.columns(3)
            for i, (key, value) in enumerate(metrics.items()):
                # Color-coded metric: green if good, red if poor
                if isinstance(value, (float, int)):
                    color = "normal" if value >= 0.8 else "orange" if value >= 0.6 else "red"
                    cols[i % 3].metric(label=key, value=round(value, 3), delta_color=color)
                else:
                    cols[i % 3].metric(label=key, value=value)

            with st.expander("What do these metrics mean?"):
                st.write("""
                    - **Accuracy**: Overall correctness  
                    - **Precision**: True positives among predicted positives  
                    - **Recall**: True positives among actual positives  
                    - **F1-Score**: Balance between precision & recall  
                    - **ROC-AUC**: Model discrimination capability  
                    - **Specificity**: True negatives among actual negatives
                """)

        # ---------------- Fairness & Bias ----------------
        with tabs[1]:
            st.subheader("⚖️ Fairness & Bias Metrics")
            if fairness:
                cols = st.columns(3)
                for i, (key, value) in enumerate(fairness.items()):
                    cols[i % 3].metric(label=key, value=round(value, 3) if isinstance(value, float) else value)
                with st.expander("Fairness explanation"):
                    st.write("""
                        - **Equal Opportunity Difference**: Difference in true positive rate across groups  
                        - **Demographic Parity Difference**: Difference in positive prediction rate across groups  
                        - **Disparate Impact Ratio**: Ratio of positive outcomes between groups
                    """)
            else:
                st.write("No fairness metrics available.")

        # ---------------- Calibration ----------------
        with tabs[2]:
            st.subheader("📈 Calibration & Reliability")
            calib_fig = evaluation.plot_calibration_curve(model_path, test_data_path)
            if calib_fig:
                st.pyplot(calib_fig, use_container_width=True)
            else:
                st.write("Calibration plot not available (model does not support probabilities).")
            with st.expander("Calibration explanation"):
                st.write("A well-calibrated model's predicted probabilities match the actual outcome frequencies.")

        # ---------------- Visualizations ----------------
        with tabs[3]:
            st.subheader("📊 Model Visualizations")

            y_true = evaluation.get_true_labels(test_data_path)
            y_pred = evaluation.get_predictions(model_path, test_data_path)
            y_prob = evaluation.get_probabilities(model_path, test_data_path)

            # Confusion Matrix
            cm_fig = evaluation.plot_confusion_matrix(y_true, y_pred)
            st.pyplot(cm_fig, use_container_width=True)
            
            # ROC Curve
            if y_prob is not None:
                roc_fig = evaluation.plot_roc_curve(y_true, y_prob)
                st.pyplot(roc_fig, use_container_width=True)

            with st.expander("Visualization explanations"):
                st.write("""
                    - **Confusion Matrix**: Shows actual vs predicted classes  
                    - **ROC Curve**: Trade-off between true positive rate & false positive rate
                """)

        # ---------------- Chatbot Advice ----------------
        with tabs[4]:
                st.subheader("💬 Model Advisor Chatbot")
                st.write("Ask for expert advice on improving your healthcare AI model.")

                user_input = st.text_input("Your question", placeholder="e.g., How to improve recall?")
                if st.button("Get Advice"):
                   advice = chatbot.get_advice(
                   st.session_state.metrics,
                   st.session_state.fairness,
                   st.session_state.operational_metrics
                   )
                   st.text_area("Expert Response", value=advice, height=300)

col1, col2 = st.columns([1,1])
if col1.button("⬅️ Back"):
    st.switch_page("pages/2_Test_Data.py")
if col2.button("Next ➡️"):
    st.switch_page("pages/4_Reports.py")