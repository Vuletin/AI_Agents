from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, session, current_app as app
from app.agents.churn_agent import MODEL_DIR, load_model, normalize_user_input, predict_churn_batch
import pandas as pd
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
            insights = aggregate_importances(feature_names, importances)[:10]

        elif model_name == "logistic_regression":
            coefs = model.named_steps["classifier"].coef_[0]
            insights = aggregate_importances(feature_names, coefs)[:10]
    
    charts_data = generate_charts_data()
    return render_template(
        "home.html",
        model_name=model_name,
        insights=insights,
        charts_data=charts_data
)


def generate_charts_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)

    # Encode churn
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    charts = {}

    # --- categorical charts (bars) ---
    categorical_features = [
        "Contract", "InternetService", "PaymentMethod", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines"
    ]
    for feature in categorical_features:
        churn_rates = df.groupby(feature)["Churn"].mean().reset_index()
        charts[feature] = {
            "type": "bar",
            "labels": churn_rates[feature].astype(str).tolist(),
            "values": (churn_rates["Churn"] * 100).round(1).tolist()
        }

        # --- numeric charts (histograms / line) ---
        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        for feature in numeric_features:
            # Convert safely to numeric
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            df[feature] = df[feature].fillna(0)

            # Bin numeric values
            bins = pd.cut(df[feature], bins=10)
            churn_rates = df.groupby(bins)["Churn"].mean().reset_index()
            charts[feature] = {
                "type": "line",
                "labels": churn_rates[feature].astype(str).tolist(),
                "values": (churn_rates["Churn"] * 100).round(1).tolist()
            }

    return charts

def aggregate_importances(feature_names, importances):
    """
    Aggregate one-hot encoded features back into their base column names.
    Example:
      - "cat__Contract_Month-to-month" → "Contract"
      - "num__MonthlyCharges" → "MonthlyCharges"
    """
    agg = {}
    for fname, importance in zip(feature_names, importances):
        if fname.startswith("cat__"):
            # cat__Feature_Category
            base = fname.split("__")[1].split("_")[0]
        elif fname.startswith("num__"):
            # num__Feature
            base = fname.split("__")[1]
        else:
            base = fname

        agg[base] = agg.get(base, 0) + abs(importance)

    # Sort descending
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

@churn_bp.route("/predict", methods=["GET", "POST"])
def predict():
    users = session.get("test_users", [])
    return render_template("predict.html", results=None, users=users)

@churn_bp.route("/predict/all", methods=["GET", "POST"])
def predict_all():
    # Support both query param (?model=logistic_regression) and form submit
    model_name = request.args.get("model") or request.form.get("model", "random_forest")
    model = load_model(model_name)

    users = session.get("test_users", [])
    if not users:
        return render_template("predict.html", results=None, users=[], model_name=model_name)

    results = predict_churn_batch(model, users)

    return render_template(
        "predict.html",
        results=results,
        users=users,
        model_name=model_name
    )

@churn_bp.route("/predict/add", methods=["POST"])
def add_user():
    raw = request.form.to_dict()
    customer_data = normalize_user_input(raw)

    # store normalized input
    if "test_users" not in session:
        session["test_users"] = []
    session["test_users"].append(customer_data)
    session.modified = True

    return redirect(url_for("churn_bp.predict"))

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