from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.special import expit
import pandas as pd
import pickle
import random
import os

# 📌 Single source of truth for schema
NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FIELDS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up from app/agents
CSV_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and normalizes input data for training and prediction."""

    numeric_fields = NUMERIC_FIELDS
    categorical_fields = CATEGORICAL_FIELDS

    df = df.copy()

    # SeniorCitizen: normalize to 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"yes": 1, "no": 0}).astype(int)

    # Normalize categorical strings
    for c in categorical_fields:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    df = df.copy()

    # SeniorCitizen: normalize to 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"yes": 1, "no": 0}).astype(int)

    # Rule 1: No internet → disable dependent services
        no_internet_mask = df["InternetService"].str.contains("no", na=False)
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
            if col in df.columns:
                df.loc[no_internet_mask, col] = "no internet service"

    # Rule 2: No phone service → MultipleLines = "no phone service"
    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        no_phone_mask = df["PhoneService"].str.contains("no", na=False)
        df.loc[no_phone_mask, "MultipleLines"] = "no phone service"

    # Rule 3: Fix TotalCharges only if missing
    if {"MonthlyCharges", "tenure", "TotalCharges"}.issubset(df.columns):
        # Ensure numeric conversion first
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        # Fill blanks with MonthlyCharges * tenure where needed
        missing_mask = df["TotalCharges"].isna()
        if missing_mask.any():
            # Use MonthlyCharges * tenure
            df.loc[missing_mask, "TotalCharges"] = (
                df.loc[missing_mask, "MonthlyCharges"].astype(float) *
                df.loc[missing_mask, "tenure"].astype(float)
            )

    # Coerce numeric fields
    for c in numeric_fields:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

def load_model(model_name="random_forest"):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def _align_df_to_pipeline(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has the exact columns and dtypes the pipeline's preprocessor expects.
    Returns a new DataFrame ready to be passed to model.predict / predict_proba.
    """
    # safety
    if "named_steps" not in dir(model) or "preprocessor" not in model.named_steps:
        raise ValueError("Model does not look like a pipeline with a 'preprocessor' step.")

    pre = model.named_steps["preprocessor"]

    # feature names the preprocessor was fitted with (original input columns order)
    # ColumnTransformer stores the original feature list here in sklearn >=1.0
    if hasattr(pre, "feature_names_in_"):
        expected_cols = list(pre.feature_names_in_)
    else:
        # fallback: try to collect columns from transformers_ tuples
        expected_cols = []
        for name, trans, cols in pre.transformers_:
            # cols can be a list of names (or slice). We'll only accept lists
            if isinstance(cols, (list, tuple)):
                expected_cols.extend(list(cols))

    # find categorical & numeric columns from the transformers_ (the third element)
    cat_cols = []
    num_cols = []
    for name, trans, cols in pre.transformers_:
        if name == "cat" or (hasattr(trans, "__class__") and "OneHotEncoder" in trans.__class__.__name__):
            # assume cols is list-like
            if isinstance(cols, (list, tuple)):
                cat_cols.extend(cols)
        if name == "num" or (hasattr(trans, "__class__") and "StandardScaler" in trans.__class__.__name__):
            if isinstance(cols, (list, tuple)):
                num_cols.extend(cols)

    # Make a copy to avoid mutating the input
    df2 = pd.DataFrame(df).copy()

    # Ensure all expected cols exist - fill with sensible defaults
    for c in expected_cols:
        if c not in df2.columns:
            # default to numeric 0 for numeric-like names, empty string for categorical
            if c in num_cols:
                df2[c] = 0
            else:
                df2[c] = "__MISSING__"

    # Coerce numeric columns
    for c in num_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)

    # Coerce categorical columns to strings and replace NaN with sentinel
    for c in cat_cols:
        if c in df2.columns:
            # replace NaN and None with sentinel and coerce to str
            df2[c] = df2[c].where(df2[c].notnull(), "__MISSING__")
            df2[c] = df2[c].astype(str).str.strip()

    # Extra safeguard: convert any column that still has mixed numeric/str to str for categorical set
    for c in expected_cols:
        # if column's dtype is object, ensure every entry is a string
        if df2[c].dtype == object:
            df2[c] = df2[c].astype(str)

    # Reorder columns to expected order (some pipelines require exact ordering)
    df2 = df2[expected_cols]

    return df2

def normalize_user_input(raw):
    user = {k: str(v).strip() for k, v in raw.items()}

    # --- PhoneService dependency ---
    if user.get("PhoneService", "").lower() == "no":
        user["MultipleLines"] = "No phone service"

    # --- InternetService dependency ---
    if user.get("InternetService", "").lower() == "no":
        for dep in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]:
            user[dep] = "No internet service"

    # SeniorCitizen fix
    if user.get("SeniorCitizen") in ["yes","Yes","1","true"]:
        user["SeniorCitizen"] = 1
    else:
        user["SeniorCitizen"] = 0

    return user

def predict_churn_batch(model, users: list):
    if not users:
        return []

    df = pd.DataFrame(users)

    # 🔹 Always sanitize before aligning
    df = sanitize_dataframe(df)

    # Align / sanitize dataframe to what the pipeline expects
    try:
        df_aligned = _align_df_to_pipeline(model, df)
    except Exception as e:
        print(f"⚠️ Alignment error: {e}")
        df_aligned = df

    # Predict churn probability
    probs = model.predict_proba(df_aligned)[:, 1]
    results = []
    for user, prob in zip(users, probs):
        # print("DEBUG RF user prob:", prob, "row:", user)
        results.append({**user, "churn_probability": round(float(prob), 4)})

    return list(map(float, probs))

def train_and_evaluate(model, data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Load raw dataset
    df = pd.read_csv(data_path)

    # Separate target
    y = df["Churn"].replace({"yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # ✅ Sanitize the DataFrame (same as predict)
    X = sanitize_dataframe(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the pipeline
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model

def train_and_save_models(CSV_PATH, rf_path, lr_path):
    print(f"Training models on: {CSV_PATH}")

    # ------------------------
    # 1. Load dataset
    # ------------------------
    df = pd.read_csv(CSV_PATH)

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Encode target
    df["Churn"] = df["Churn"].map({"No": 0, "yes": 1})

    # ------------------------
    # 2. Sanitize consistently
    # ------------------------
    df = sanitize_dataframe(df)

    # Split features/target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Categorical & numeric cols
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    # ------------------------
    # 3. Train/test split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------
    # 4. Pipelines
    # ------------------------
    # Random Forest (calibrated)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    calibrated_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=5)

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_rf)
    ])

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    # ------------------------
    # 5. Train
    # ------------------------
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    # ------------------------
    # 6. Evaluate
    # ------------------------
    rf_score = rf_pipeline.score(X_test, y_test)
    lr_score = lr_pipeline.score(X_test, y_test)

    print("✅ Models trained and saved!")
    print(f"   - RF score: {rf_score:.3f}")
    print(f"   - LR score: {lr_score:.3f}")

    # Optional: quick sanity check on churn distribution
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    print(f"   - RF avg churn probability (churners): {rf_probs[y_test == 1].mean():.3f}")
    print(f"   - RF avg churn probability (non-churners): {rf_probs[y_test == 0].mean():.3f}")

    # ------------------------
    # 7. Save
    # ------------------------
    with open(rf_path, "wb") as f:
        pickle.dump(rf_pipeline, f)

    with open(lr_path, "wb") as f:
        pickle.dump(lr_pipeline, f)

    return {
        "rf_score": rf_score,
        "lr_score": lr_score
    }

def churn_summary(CSV_PATH):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    summary = {}
    
    # Confirm CSV loading
    print("Loading CSV from:", CSV_PATH)
    print("First 5 rows:\n", df.head())

    # Detect categorical columns
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and col not in ['customerID', 'Churn']
    ]
    print("Categorical columns detected:", categorical_cols)

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df['Churn'])
        # Ensure both 'yes' and 'No' exist
        for churn_val in ['No', 'yes']:
            if churn_val not in ct.columns:
                ct[churn_val] = 0
        ct = ct[['No', 'yes']]

        # Convert to list of dicts
        summary[col] = [
            {
                "value": idx,
                "Stayed (No Churn)": int(row['No']),
                "Churned": int(row['yes'])
            }
            for idx, row in ct.iterrows()
        ]

    # Final confirmation before returning
    print("Final summary dictionary keys:", list(summary.keys()))
    return summary

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

def churn_by_category_json():

    df = pd.read_csv(CSV_PATH)
    charts_data = {}   # ✅ must initialize here once

    # Convert SeniorCitizen from 0/1 to No/yes
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "yes"})

    # Handle categorical features
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and col not in ['customerID', 'Churn']
    ]

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        if 'yes' not in ct.columns:  # safety check
            ct['yes'] = 0
        churn_rates = ct['yes']
        charts_data[col] = {
            "labels": ct.index.tolist(),
            "values": churn_rates.round(1).tolist(),
            "type": "bar"
        }

    # Handle continuous features
    for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
        if col in df.columns:
            # Force numeric conversion (coerce errors to NaN)
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop NaNs to avoid issues
            df_clean = df.dropna(subset=[col, "Churn"]).copy()

            if not df_clean.empty:
                df_sorted = df_clean.sort_values(col)
                bins = pd.cut(df_sorted[col], bins=20)  # group into 20 bins
                churn_rate = df_sorted.groupby(bins)['Churn'].apply(
                    lambda x: (x == 'yes').mean() * 100
                )
                charts_data[col] = {
                    "labels": [f"{interval.left:.0f}-{interval.right:.0f}" for interval in churn_rate.index],
                    "values": churn_rate.round(1).tolist(),
                    "type": "line"
                }
                
    return charts_data

def explain_single_logistic(model, user_row):
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # transform row with same preprocessing
    Xt = pre.transform(user_row)
    feature_names = pre.get_feature_names_out()

    # coefficients
    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]

    contributions = Xt.toarray()[0] * coefs  # each feature’s contribution
    logit = intercept + contributions.sum()
    prob = expit(logit)

    print(f"Intercept: {intercept:.4f}")
    print(f"Predicted churn probability: {prob:.4f}")
    for name, contrib in sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name}: {contrib:.4f}")

def random_user():
    user = {}

    user["gender"] = random.choice(["Male", "Female"])
    user["SeniorCitizen"] = random.choice([0, 1])
    user["Partner"] = random.choice(["Yes", "No"])
    user["Dependents"] = random.choice(["Yes", "No"])
    user["tenure"] = random.randint(0, 72)
    
    # PhoneService and MultipleLines
    phone = random.choice(["Yes", "No"])
    user["PhoneService"] = phone
    if phone == "No":
        user["MultipleLines"] = "No phone service"
    else:
        user["MultipleLines"] = random.choice(["Yes", "No"])

    # InternetService and its dependents
    internet = random.choice(["DSL", "Fiber optic", "No"])
    user["InternetService"] = internet
    if internet == "No":
        # force all dependents to "No internet service"
        for dep in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]:
            user[dep] = "No internet service"
    else:
        for dep in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]:
            user[dep] = random.choice(["Yes", "No"])

    user["Contract"] = random.choice(["Month-to-month", "One year", "Two year"])
    user["PaperlessBilling"] = random.choice(["Yes", "No"])
    user["PaymentMethod"] = random.choice([
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    user["MonthlyCharges"] = round(random.uniform(18.0, 120.0), 2)
    user["TotalCharges"] = round(user["MonthlyCharges"] * user["tenure"], 2)

    return user