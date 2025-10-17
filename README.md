<div align="center">

# 🧠 Churn Predict  
### Customer Churn Prediction Web App  

🎯 *Predict which customers are likely to leave — powered by Logistic Regression and Random Forest models.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/status-Active-success)

</div>

---

## 🚀 Overview

**Churn Predict** is a machine learning web application that analyzes customer data and predicts churn probability in real time.  
It combines two powerful models — **Logistic Regression** and **Random Forest** — optimized and tuned for balanced accuracy and interpretability.  

The app is designed to help telecom and subscription-based businesses **identify customers at risk of leaving** and make data-driven retention decisions.

---

## 🧩 Key Features

- **Real-Time Prediction** — enter customer info and instantly get churn probability from both models.  
- **Dual-Model Comparison** — view how Logistic Regression and Random Forest differ in decision behavior.  
- **Dynamic Dashboard** — visualize churn by categories like contract, payment type, and internet service.  
- **Sortable Data Table** — neatly styled with zebra rows, hover highlights, and rounded corners.  
- **Random Data Generator** — quickly test model behavior with randomized user samples.  

---

## 🤖 Machine Learning Details

Both models were trained on **7,000 users from the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)** on Kaggle.

| Model | Strength | Description |
|--------|-----------|--------------|
| **Logistic Regression** | 🧮 Interpretability | Provides transparent probability estimates useful for understanding feature influence. |
| **Random Forest** | 🌲 Accuracy & Stability | Handles non-linear relationships and feature interactions to improve predictive power. |

During optimization, I tuned hyperparameters, applied feature encoding & scaling, and compared ROC-AUC and F1-scores to ensure both models perform robustly on unseen data.

---

## 🧰 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend** | Flask, scikit-learn, pandas, joblib |
| **Frontend** | HTML, CSS, Chart.js |
| **Data** | 7,000 Telco Customer records (Kaggle) |
| **Deployment** | Flask (local or Render) |

---

## 🧭 Folder Structure

AI_Agents/
├── app
│   ├── agents
│   │   ├── automation_agent.py
│   │   ├── churn_agent.py
│   │   ├── __init__.py
│   │   └── sentiment_agent.py
│   ├── constants.py
│   ├── data
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── churn_model.pkl
│   │   ├── logistic_regression.pkl
│   │   └── random_forest.pkl
│   ├── routes
│   │   ├── churn_routes.py
│   │   └── __init__.py
│   ├── static
│   │   ├── favicon.ico
│   │   └── styles.css
│   └── templates
│       ├── base.html
│       ├── dashboard.html
│       ├── home.html
│       ├── login.html
│       ├── predict.html
│       └── upload.html
├── Dockerfile
├── __init__.py
├── README.md
└── requirements.txt

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-predict.git
cd churn-predict

# Create virtual environment
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
Then open:
👉 http://127.0.0.1:5000

🧭 Roadmap

 Logistic Regression & Random Forest Models

 Interactive Dashboard with Charts

 Random User Generator

 Export table & charts to CSV

 Upload custom CSV for batch predictions

 Evolve into a General ML Prediction App where users can:

Upload any dataset

Automatically train a model

Test predictions on new samples

📸 Screenshots

(Coming soon — dashboard and prediction view)

<div align="center">
🧾 License

Released under the MIT License

💡 Built with Python, Flask, and a lot of curiosity about machine learning.

</div> ```

