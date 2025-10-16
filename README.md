## Churn Agent

## 🧠 Churn Predict – Customer Churn Prediction Web App

Churn Predict is a full-featured machine learning web application built with Flask for predicting customer churn based on demographic and service data.
It provides a clean dashboard for visualizing churn trends, managing users, and testing different prediction models in real time.

## 🚀 Features
## 🧩 Core Functionality

User Input Form – enter customer details (gender, tenure, contract type, monthly charges, etc.) and instantly get churn prediction.

Random User Generator – automatically fill the form with randomized customer data for quick testing.

Machine Learning Models – supports multiple trained models (Random Forest, Logistic Regression, etc.) with probability outputs.

Detailed Prediction Output – shows churn probability and predicted outcome.

## 📊 Dashboard & Visualization

Interactive Charts – visual breakdown of churn by contract, payment method, internet service, and more.

Dynamic Table – view all predictions made during the session with sortable, zebra-styled, hover-highlighted rows.

Data Insights – real churn statistics computed directly from the original dataset.

## 🧰 Data & Backend

Clean Data Pipeline – automatic normalization of categorical/numerical inputs for consistent predictions.

Reusable ML Pipeline – trained models are saved and loaded via joblib for fast inference.

Logging & Debugging Tools – print logs of model scores and inputs for transparency.

## 🧪 Tech Stack
Layer	Technologies
Backend	Flask, scikit-learn, pandas, joblib
Frontend	HTML, CSS, Bootstrap, Chart.js
Data	CSV dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Deployment	Localhost (Flask), easily adaptable to Render / Railway / Heroku
## 🧭 Folder Structure
AI_Agents/
│
├── app/
│   ├── routes/
│   │   └── churn_routes.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── dashboard.html
│   │   └── predict.html
│   └── __init__.py
│
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── main.py
└── requirements.txt

🧑‍💻 How to Run Locally

Clone the repository

git clone https://github.com/yourusername/churn-predict.git
cd churn-predict


Create virtual environment

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)


Install dependencies

pip install -r requirements.txt


Run the app

python main.py


Open in browser

http://127.0.0.1:5000

📈 Future Roadmap

✅ Current: Fully functional churn prediction and analytics dashboard
🧩 Planned:

Export chart & prediction table to CSV

Upload custom CSV dataset for batch prediction

Build a general ML prediction app where users can:

Upload any dataset

Automatically train a model

Use a “Predict” page to test new samples

📷 Screenshots (Optional)

(Add images of dashboard, form, and chart views here)

🧾 License

This project is open source and available under the MIT License
.
