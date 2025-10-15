import random
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, session, current_app as app
from app.agents.churn_agent import MODEL_DIR, churn_by_category_json, load_model, normalize_user_input, predict_churn_batch, aggregate_importances, random_user
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

churn_bp = Blueprint('churn_bp', __name__)

# Simple in-memory server-side store for test users to avoid blowing up the client-side session cookie
# Keyed by a small UUID stored in the user's session. This is ephemeral and per-process.
USER_STORE = {}

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
    # Use server-side store keyed by a UUID in the session
    store_id = session.get('user_store_id')
    users = USER_STORE.get(store_id, []) if store_id else []

    rf_results = []
    lr_results = []
    # If there are users in session, compute predictions so the chart can render
    if users:
        try:
            rf_model = load_model("random_forest")
            lr_model = load_model("logistic_regression")
            rf_results = predict_churn_batch(rf_model, users) or []
            lr_results = predict_churn_batch(lr_model, users) or []
        except FileNotFoundError:
            # models not trained/saved yet
            rf_results = []
            lr_results = []

    return render_template(
        "predict.html",
        users=users,
        rf_results=rf_results,
        lr_results=lr_results
    )

@churn_bp.route("/predict/all", methods=["GET", "POST"])
def predict_all():
    store_id = session.get('user_store_id')
    users = USER_STORE.get(store_id, []) if store_id else []
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
    is_random = "random" in request.form
    # Ensure a store_id exists for this session
    store_id = session.get('user_store_id')
    if not store_id:
        store_id = str(uuid.uuid4())
        session['user_store_id'] = store_id

    if store_id not in USER_STORE:
        USER_STORE[store_id] = []

    # Case 1: Normal manual input from form
    if count == 1 and not is_random:
        raw = request.form.to_dict()
        raw.pop("count", None)  # remove button value
        raw.pop("random", None)  # safety: also remove hidden random field if any
        customer_data = normalize_user_input(raw)
        USER_STORE[store_id].append(customer_data)

    # Case 2: Random users (bulk or +1 random button)
    else:
        for _ in range(count):
            user = random_user()
            user["tenure"] = int(user.get("tenure", random.randint(1, 72)))
            user["MonthlyCharges"] = float(user.get("MonthlyCharges", round(random.uniform(18.0, 120.0), 2)))
            user["TotalCharges"] = round(user["MonthlyCharges"] * user["tenure"], 2)
            customer_data = normalize_user_input(user)
            USER_STORE[store_id].append(customer_data)

    session.modified = True

    rf_model = load_model("random_forest")
    lr_model = load_model("logistic_regression")

    users = USER_STORE.get(session.get('user_store_id'), [])
    rf_results = predict_churn_batch(rf_model, users)
    lr_results = predict_churn_batch(lr_model, users)

    rf_results = rf_results if rf_results else []
    lr_results = lr_results if lr_results else []

    return render_template(
        "predict.html",
        users=users,
        rf_results=rf_results,
        lr_results=lr_results,
    )

@churn_bp.route('/predict/clear', methods=['POST'])
def clear_users():
    store_id = session.pop('user_store_id', None)
    if store_id and store_id in USER_STORE:
        USER_STORE.pop(store_id, None)
    return redirect(url_for("churn_bp.predict"))

@churn_bp.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )