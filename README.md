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
```text
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
```

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
git clone https://github.com/yourusername/churn-predict.git
cd churn-predict

python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)

pip install -r requirements.txt

python main.py

Then open in your browser:
👉 http://127.0.0.1:5000

---

## 🧭 Roadmap

- [x] Logistic Regression & Random Forest Models  
- [x] Interactive Dashboard with Charts  
- [x] Random User Generator  
- [ ] Export table & charts to **CSV**  
- [ ] Upload custom CSV for **batch predictions**  
- [ ] Evolve into a **General ML Prediction App**, where users can:  
  - Upload any dataset  
  - Automatically train a model  
  - Test predictions on new samples

## 📸 Screenshots
*(Coming soon — dashboard and prediction view)*

<div align="center">

## 🧑‍💻 About the Author

**Sava** is a self-taught developer with a lifelong passion for computers and technology.  
Focusing on **full-stack development** and **AI applications**.  
He has already built **senior-level projects**, including:

- 🏦 A **Finance/Banking App** built with Flask and SQLite  
- 🧠 **Churn Predict**, a machine learning app using Random Forest & Logistic Regression  
- 🤖 **AI_Agents**, a growing suite of AI tools including:
  - **Churn Predict**
  - **Sentiment Analysis**
  - **Automation Agent**

He plans to **dockerize and deploy** all AI_Agents on **Google Cloud**, creating a unified AI platform.  
Sava is also exploring **Linux systems**, **low-level programming in C**, and **Python-based AI** integrated with **camera apps** and **LLMs**.

## 🧾 License

Released under the [MIT License](LICENSE)

💡 Built with **Flask**, **scikit-learn**, and a lifelong curiosity for **machine learning**.



