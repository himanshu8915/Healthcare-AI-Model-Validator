from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ---------------- Memory Setup ----------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- Prompt Template ----------------
PROMPT_TEMPLATE = """
You are a healthcare AI expert assisting in validating diagnostic models.

The user has provided the following metrics:

{input_text}

Provide detailed guidance on:

1. How to improve model performance.
2. Possible reasons for any metric being low.
3. How to handle biases if fairness metrics indicate issues.
4. Calibration and reliability improvements.
5. Any healthcare-specific considerations for this model.

Respond concisely, clearly, and in actionable steps.
"""

prompt = PromptTemplate(
    input_variables=["input_text"],
    template=PROMPT_TEMPLATE
)

# ---------------- Initialize MedGemma LLM ----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------------- Create LLMChain ----------------
expert_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_key="text"  # ensures invoke() returns a dict with key 'text'
)

# ---------------- Advice Function ----------------
def get_advice(performance_metrics, fairness_metrics, operational_metrics=None):
    """
    Return expert healthcare advice based on model evaluation metrics.

    Args:
        performance_metrics (dict): e.g., {"Accuracy": 0.92, "F1-Score": 0.88}
        fairness_metrics (dict): e.g., {"Male": 0.95, "Female": 0.93}
        operational_metrics (dict): Optional operational metrics

    Returns:
        str: Expert advice response
    """
    # Convert all metrics to one string
    perf_str = "\n".join([f"{k}: {v}" for k, v in performance_metrics.items()])
    fairness_str = "\n".join([f"{k}: {v}" for k, v in fairness_metrics.items()])
    operational_str = ""
    if operational_metrics:
        operational_str = "\n".join([f"{k}: {v}" for k, v in operational_metrics.items()])

    metrics_str = f"Performance Metrics:\n{perf_str}\n\nFairness Metrics:\n{fairness_str}"
    if operational_metrics:
        metrics_str += f"\n\nOperational Metrics:\n{operational_str}"

    # Pass as single input_text key
    response = expert_chain.invoke({
        "input_text": metrics_str
    })

    return response["text"]  # invoke() returns a dict with output key 'text'