# 🏥 Healthcare AI Model Validator

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Workflow](#workflow)
* [Chatbot Advisor](#chatbot-advisor)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Overview

The **Healthcare AI Model Validator** is a **web application** built with **Streamlit** that helps healthcare ML engineers and data scientists **validate, evaluate, and improve diagnostic models**. It provides a **step-by-step workflow** to upload models, generate or upload test data, evaluate performance and fairness, visualize metrics, download reports, and interact with an **expert healthcare AI chatbot** for guidance.

The goal is to make model validation **transparent, structured, and actionable**, specifically for healthcare applications where fairness and reliability are crucial.

---

## Features

1. **Step-by-Step Workflow**

   * **Model Upload:** Upload pre-trained ML models (`.pkl` format).
   * **Test Data Upload / Generation:** Upload your dataset or generate synthetic test data on the fly.
   * **Evaluation Metrics:** Includes:

     * **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Calibration.
     * **Fairness Metrics:** Equal Opportunity, Demographic Parity, Disparate Impact.
     * **Operational Metrics:** Latency, resource usage, throughput.
   * **Visualizations:** Confusion matrix, ROC curve, calibration plots.
   * **Report Download:** Generate PDF reports summarizing all metrics and insights.
   * **Chatbot Advisor:** MedGemma-powered AI assistant provides healthcare-specific advice.

2. **Interactive Chatbot**

   * Uses evaluation metrics to give **actionable guidance**.
   * Explains low metrics, suggests improvements, and highlights biases.
   * Provides advice on **calibration, reliability, and healthcare considerations**.

3. **User-Friendly Dashboard**

   * Multi-page interface for smooth navigation.
   * Interactive charts and tables for easy interpretation.
   * Automatic state management with **Streamlit session state**.

---

## Project Structure

```
SWE_Project/
│
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ backend/               # Backend modules
│   ├─ chatbot.py         # Healthcare AI chatbot integration
│   ├─ evaluation.py      # Evaluation metrics and plots
│   ├─ data_generation.py # Synthetic test data generation
│   └─ utils.py           # Helper functions
├─ pages/                 # Streamlit multi-page workflow
│   ├─ 1_Model_Upload.py
│   ├─ 2_Test_Data.py
│   ├─ 3_Evaluation.py
│   ├─ 4_Reports.py
│   └─ 5_Chatbot.py
├─ temp_data/             # Temporary test datasets
├─ temp_models/           # Temporary model storage
└─ evaluation_report.pdf  # Generated PDF report
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/himanshu8915/Healthcare-AI-Model-Validator.git
cd Healthcare-AI-Model-Validator
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables (for Chatbot API keys):

* Create a `.env` file in the project root.
* Add your keys:

```env
GOOGLE_API_KEY=<your_google_api_key>
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app contains **5 pages**:

1. **Model Upload**
2. **Test Data Upload / Generation**
3. **Evaluation Metrics & Visualizations**
4. **Report Download**
5. **Healthcare Chatbot Advisor**

---

## Workflow

1. **Upload Your Model:** `.pkl` or supported format.
2. **Upload / Generate Test Data:** Option to upload CSV or generate synthetic data.
3. **Evaluate Model:** View performance, fairness, and operational metrics.
4. **Visualize Metrics:** Confusion matrix, ROC curve, calibration plots.
5. **Download Report:** PDF summarizing all metrics.
6. **Chat with Advisor:** Ask questions about model improvement, bias mitigation, or calibration.

---

## Chatbot Advisor

* Powered by **MedGemma (Google Gemini)** via **LangChain**.
* Uses **all model evaluation metrics** to provide guidance.
* Offers:

  * Tips to improve performance.
  * Bias detection and mitigation.
  * Healthcare-specific insights.
  * Calibration and reliability suggestions.

---

## Dependencies

* **Streamlit** – Interactive web framework
* **LangChain** – Chatbot orchestration
* **Google GenAI / MedGemma** – LLM for expert guidance
* **scikit-learn** – Model evaluation metrics
* **pandas / numpy** – Data handling
* **matplotlib / seaborn / plotly** – Visualizations
* **reportlab** – PDF report generation

---

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add feature"`
4. Push branch: `git push origin feature-name`
5. Open a Pull Request

---

## License

MIT License – Open-source for research and development in healthcare AI.

---

## Contact

* **GitHub:** [himanshu8915](https://github.com/himanshu8915)
* **Email:** [himanshusharma14024@gmail.com](mailto:himanshusharma14024@gmail.com)

---

## Screenshots (Optional)

*You can add screenshots of your app interface for better clarity:*

* Model Upload Page
* Test Data Upload / Generation
* Evaluation Metrics Dashboard
* Chatbot Advisor Interaction
* PDF Report P
