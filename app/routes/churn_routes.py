from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, session, current_app as app
from app.agents.churn_agent import MODEL_DIR, churn_by_category_json, load_model, normalize_user_input, predict_churn_batch, aggregate_importances, random_user
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

churn_bp = Blueprint('churn_bp', __name__)

try:
    model = load_model("random_forest")  # default to RF
except FileNotFoundError:
    model = None

@churn_bp.route("/")
def home():
    model_name = request.args.get("model", "random_forest")
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        model = None

    insights = None
    if model:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        if model_name == "random_forest":
            importances = model.named_steps["classifier"].feature_importances_
            insights = aggregate_importances(feature_names, importances)

        elif model_name == "logistic_regression":
            coefs = model.named_steps["classifier"].coef_[0]
            insights = aggregate_importances(feature_names, coefs)
    
    # 👇 call the churn_agent function with a CSV path
    charts_data = churn_by_category_json()
    print("DEBUG charts_data keys passed to template:", list(charts_data.keys()))

    return render_template(
        "home.html",
        model_name=model_name,
        insights=insights,
        charts_data=charts_data
    )

@churn_bp.route("/predict", methods=["GET", "POST"])
def predict():
    users = session.get("test_users", [])
    return render_template(
        "predict.html",
        users=users,
        rf_results=[],
        lr_results=[]
    )

@churn_bp.route("/predict/all", methods=["GET", "POST"])
def predict_all():
    users = session.get("test_users", [])
    if not users:
        return render_template("predict.html", users=[], rf_results=[], lr_results=[])

    rf_model = load_model("random_forest")
    lr_model = load_model("logistic_regression")

    rf_results = predict_churn_batch(rf_model, users)
    lr_results = predict_churn_batch(lr_model, users)

    return render_template(
        "predict.html",
        users=users,
        rf_results=rf_results,
        lr_results=lr_results
    )

@churn_bp.route("/predict/add", methods=["POST"])
def add_user():
    count = int(request.form.get("count", "1"))

    if "test_users" not in session:
        session["test_users"] = []

    if count == 1:
        raw = request.form.to_dict()
        customer_data = normalize_user_input(raw)
        session["test_users"].append(customer_data)
    else:
        for _ in range(count):
            user = random_user()
            user["TotalCharges"] = round(user["MonthlyCharges"] * user["tenure"], 2)
            customer_data = normalize_user_input(user)
            session["test_users"].append(customer_data)

    session.modified = True

    # ✅ Predict with both models
    users = session.get("test_users", [])
    rf_model = load_model("random_forest")
    lr_model = load_model("logistic_regression")

    rf_results = predict_churn_batch(rf_model, users)
    lr_results = predict_churn_batch(lr_model, users)

    return render_template(
        "predict.html",
        users=users,
        rf_results=rf_results,
        lr_results=lr_results,
    )

@churn_bp.route('/predict/clear', methods=['POST'])
def clear_users():
    session.pop("test_users", None)
    return redirect(url_for("churn_bp.predict"))

@churn_bp.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )