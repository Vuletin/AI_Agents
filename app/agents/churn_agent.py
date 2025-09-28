from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
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

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and normalizes input data for training and prediction."""

    numeric_fields = NUMERIC_FIELDS
    categorical_fields = CATEGORICAL_FIELDS

    df = df.copy()

    # SeniorCitizen: normalize to 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0}).astype(int)

    # Normalize categorical strings
    for c in categorical_fields:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    df = df.copy()

    # SeniorCitizen: normalize to 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0}).astype(int)

    # Rule 1: No internet → disable dependent services
    if "InternetService" in df.columns:
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
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill blanks with MonthlyCharges * tenure
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = (
            df["MonthlyCharges"].astype(float) * df["tenure"].astype(float)
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
    """
    Ensure user input matches the training schema:
    - numeric fields → float or int
    - categorical fields → strings
    - fill missing values safely
    """
    customer_data = {}

    # numeric fields
    try:
        customer_data["TotalCharges"] = float(raw.get("TotalCharges", 0) or 0.0)
    except ValueError:
        customer_data["TotalCharges"] = 0.0

    try:
        customer_data["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0) or 0.0)
    except ValueError:
        customer_data["MonthlyCharges"] = 0.0

    try:
        customer_data["tenure"] = int(raw.get("tenure", 0) or 0)
    except ValueError:
        customer_data["tenure"] = 0

    try:
        customer_data["SeniorCitizen"] = int(raw.get("SeniorCitizen", 0) or 0)
    except ValueError:
        customer_data["SeniorCitizen"] = 0

    # categorical fields
    categorical_fields = CATEGORICAL_FIELDS

    for field in categorical_fields:
        val = raw.get(field, "")
        if val is None or str(val).strip() == "":
            customer_data[field] = "Unknown"   # placeholder category
        else:
            customer_data[field] = str(val).strip().lower()

    return customer_data

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
        results.append({**user, "churn_probability": round(float(prob), 4)})

    print("DEBUG sanitized user row:", df.iloc[0].to_dict())
    return results

def train_and_evaluate(model, data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Load raw dataset
    df = pd.read_csv(data_path)

    # Separate target
    y = df["Churn"].replace({"Yes": 1, "No": 0})
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
    
def train_and_save_models(csv_path, rf_path, lr_path):
    print(f"Training models on: {csv_path}")

    # ------------------------
    # 1. Load dataset
    # ------------------------
    df = pd.read_csv(csv_path)

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Encode target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

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

def churn_summary(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    summary = {}
    
    # Confirm CSV loading
    print("Loading CSV from:", csv_path)
    print("First 5 rows:\n", df.head())

    # Detect categorical columns
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and col not in ['customerID', 'Churn']
    ]
    print("Categorical columns detected:", categorical_cols)

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df['Churn'])
        # Ensure both 'Yes' and 'No' exist
        for churn_val in ['No', 'Yes']:
            if churn_val not in ct.columns:
                ct[churn_val] = 0
        ct = ct[['No', 'Yes']]

        # Convert to list of dicts
        summary[col] = [
            {
                "value": idx,
                "Stayed (No Churn)": int(row['No']),
                "Churned": int(row['Yes'])
            }
            for idx, row in ct.iterrows()
        ]

    # Final confirmation before returning
    print("Final summary dictionary keys:", list(summary.keys()))
    return summary

def generate_charts_data(csv_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(csv_path)

    charts = {}

    # Contract (bar chart)
    charts["Contract"] = {
        "type": "bar",
        "labels": df["Contract"].unique().tolist(),
        "values": df.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()*100).tolist()
    }

    # InternetService (bar chart)
    charts["InternetService"] = {
        "type": "bar",
        "labels": df["InternetService"].unique().tolist(),
        "values": df.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()*100).tolist()
    }

    # Tenure (line chart, binned)
    df["tenure_bin"] = pd.cut(df["tenure"], bins=10)
    charts["tenure"] = {
        "type": "line",
        "labels": df["tenure_bin"].astype(str).unique().tolist(),
        "values": df.groupby("tenure_bin")["Churn"].apply(lambda x: (x=="Yes").mean()*100).tolist()
    }

    return charts

def churn_by_category_json(csv_path):

    df = pd.read_csv(csv_path)
    charts_data = {}   # ✅ must initialize here once

    # Convert SeniorCitizen from 0/1 to No/Yes
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Handle categorical features
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and col not in ['customerID', 'Churn']
    ]

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        if 'Yes' not in ct.columns:  # safety check
            ct['Yes'] = 0
        churn_rates = ct['Yes']
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
                    lambda x: (x == 'Yes').mean() * 100
                )
                charts_data[col] = {
                    "labels": [f"{interval.left:.0f}-{interval.right:.0f}" for interval in churn_rate.index],
                    "values": churn_rate.round(1).tolist(),
                    "type": "line"
                }

    return charts_data