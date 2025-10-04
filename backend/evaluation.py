import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(model_path: str, test_data_path: str) -> dict:
    """
    Evaluate a classification model on a test dataset.
    
    Args:
        model_path: Path to the saved model (.pkl)
        test_data_path: Path to the test dataset CSV
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # --- Load test data ---
    df = pd.read_csv(test_data_path)
    
    if 'label' not in df.columns:
        raise ValueError("Test dataset must contain a 'label' column for true targets")
    
    X_test = df.drop(columns=['label'])
    y_test = df['label']
    
    # --- Load model ---
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # --- Predict ---
    y_pred = model.predict(X_test)
    
    # --- Compute metrics ---
    metrics = {}
    
    metrics['Accuracy'] = round(accuracy_score(y_test, y_pred), 3)
    metrics['Precision'] = round(precision_score(y_test, y_pred, zero_division=0), 3)
    metrics['Recall'] = round(recall_score(y_test, y_pred, zero_division=0), 3)
    metrics['F1-Score'] = round(f1_score(y_test, y_pred, zero_division=0), 3)
    
    # ROC-AUC requires probability predictions
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['ROC-AUC'] = round(roc_auc_score(y_test, y_prob), 3)
    except:
        metrics['ROC-AUC'] = "N/A (model does not support predict_proba)"
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['Specificity'] = round(tn / (tn + fp), 3)
    
    return metrics
def get_true_labels(test_data_path):
    import pandas as pd
    df = pd.read_csv(test_data_path)
    return df['label']

def get_predictions(model_path, test_data_path):
    import pickle, pandas as pd
    df = pd.read_csv(test_data_path)
    X = df.drop(columns=['label'])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model.predict(X)

def get_probabilities(model_path, test_data_path):
    import pickle, pandas as pd
    df = pd.read_csv(test_data_path)
    X = df.drop(columns=['label'])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    try:
        return model.predict_proba(X)[:, 1]
    except:
        return None
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

# --- Existing functions: evaluate_model, get_true_labels, get_predictions, get_probabilities ---

def compute_fairness_metrics(model_path, test_data_path, sensitive_column="gender") -> dict:
    """
    Compute basic fairness metrics for a binary classification model.
    
    Args:
        model_path: Path to the model (.pkl)
        test_data_path: Path to CSV test data
        sensitive_column: Column name to check fairness (e.g., "gender")
        
    Returns:
        fairness_metrics: Dictionary
    """
    df = pd.read_csv(test_data_path)
    
    if sensitive_column not in df.columns:
        return {"Note": f"No {sensitive_column} column in dataset"}
    
    X = df.drop(columns=['label'])
    y_true = df['label']
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X)
    
    groups = df[sensitive_column].unique()
    metrics = {}
    
    if len(groups) != 2:
        return {"Note": "Fairness metrics currently support 2 groups only"}
    
    g1, g2 = groups
    mask1 = df[sensitive_column] == g1
    mask2 = df[sensitive_column] == g2
    
    # Positive prediction rates
    ppr1 = y_pred[mask1].mean()
    ppr2 = y_pred[mask2].mean()
    
    # True positive rates
    tpr1 = (y_pred[mask1] & y_true[mask1]).sum() / (y_true[mask1].sum() + 1e-6)
    tpr2 = (y_pred[mask2] & y_true[mask2]).sum() / (y_true[mask2].sum() + 1e-6)
    
    metrics["Demographic Parity Difference"] = round(abs(ppr1 - ppr2), 3)
    metrics["Equal Opportunity Difference"] = round(abs(tpr1 - tpr2), 3)
    metrics["Disparate Impact Ratio"] = round(ppr1 / (ppr2 + 1e-6), 3)
    
    return metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

def plot_calibration_curve(model_path, test_data_path):
    """
    Plots a calibration curve (reliability diagram)
    """
    import pandas as pd
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(test_data_path)
    X = df.drop(columns=['label'])
    y_true = df['label']
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except:
        return None  # Model does not support predict_proba
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o', label='Calibration')
    ax.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Calibration Curve")
    ax.legend()
    return fig
import time
import psutil
import os

def compute_operational_metrics(model, test_data_path):
    import pandas as pd
    import joblib

    # Load model
    if isinstance(model, str):
        model = joblib.load(model)
    
    # Load test data
    df = pd.read_csv(test_data_path)
    X = df.drop(columns=['label'])

    # Measure inference latency
    start_time = time.time()
    model.predict(X)
    latency = time.time() - start_time
    avg_latency_ms = latency / len(X) * 1000  # ms per sample

    # Throughput: samples per second
    throughput = len(X) / latency

    # Resource usage: approximate memory usage
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "Avg Inference Latency (ms/sample)": round(avg_latency_ms, 3),
        "Throughput (samples/sec)": round(throughput, 2),
        "Memory Usage (MB)": round(mem_usage_mb, 2)
    }

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig

# backend/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true (list or np.array): True labels.
        y_prob (list or np.array): Predicted probabilities for the positive class.
    
    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    
    return fig

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd

def plot_calibration_curve(model_path, test_data_path, n_bins=10):
    """
    Plot calibration curve for model probabilities.
    """
    import joblib
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    if "label" not in df.columns:
        return None
    
    X = df.drop(columns=["label"])
    y_true = df["label"].values
    
    # Check if model supports predict_proba
    if not hasattr(model, "predict_proba"):
        return None
    
    y_prob = model.predict_proba(X)[:,1]
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(prob_pred, prob_true, marker='o', label="Calibration")
    ax.plot([0,1],[0,1], linestyle='--', color='gray', label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    
    return fig

import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_report(metrics, fairness, operational=None, filename="evaluation_report.pdf"):
    """
    Generate evaluation report using reportlab.
    Args:
        metrics (dict): Performance metrics {name: value}
        fairness (dict): Fairness metrics {group: bias_score}
        operational (dict): Optional operational metrics {metric: value}
        filename (str): Output PDF filename
    Returns:
        str: Path to generated PDF
    """
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("📑 Healthcare Model Evaluation Report", styles['Title']))
    elements.append(Spacer(1, 16))

    # --- Performance Metrics ---
    elements.append(Paragraph("📊 Performance Metrics", styles['Heading2']))
    metrics_data = [["Metric", "Value"]]
    for k, v in metrics.items():
        metrics_data.append([k, str(v)])

    perf_table = Table(metrics_data, hAlign="LEFT")
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 12))

    # --- Fairness Metrics ---
    elements.append(Paragraph("⚖️ Fairness Metrics", styles['Heading2']))
    fairness_data = [["Group", "Bias Score"]]
    for group, score in fairness.items():
        fairness_data.append([group, str(score)])

    fair_table = Table(fairness_data, hAlign="LEFT")
    fair_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(fair_table)
    elements.append(Spacer(1, 12))

    # --- Operational Metrics (optional) ---
    if operational:
        elements.append(Paragraph("⚙️ Operational Metrics", styles['Heading2']))
        op_data = [["Metric", "Value"]]
        for k, v in operational.items():
            op_data.append([k, str(v)])

        op_table = Table(op_data, hAlign="LEFT")
        op_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        elements.append(op_table)
        elements.append(Spacer(1, 12))

    # --- Footer ---
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("✅ Report generated successfully using ReportLab", styles['Normal']))

    # Build PDF
    doc.build(elements)
    return os.path.abspath(filename)
