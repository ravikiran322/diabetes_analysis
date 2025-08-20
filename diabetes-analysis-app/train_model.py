import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import urllib.request

DATA_PATH = os.path.join("data", "diabetes.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "feature_stats.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# A known mirror of the PIMA dataset on GitHub (raw CSV).
RAW_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
]

def ensure_dataset():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs("data", exist_ok=True)
    print("Dataset not found. Attempting download...")
    last_err = None
    for url in RAW_URLS:
        try:
            if url.endswith("pima-indians-diabetes.data.csv"):
                # That file has no header; define columns per PIMA schema
                cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
                tmp_path = DATA_PATH + ".tmp"
                urllib.request.urlretrieve(url, tmp_path)
                df = pd.read_csv(tmp_path, header=None, names=cols)
                df.to_csv(DATA_PATH, index=False)
                os.remove(tmp_path)
            else:
                urllib.request.urlretrieve(url, DATA_PATH)
            print(f"Downloaded dataset from: {url}")
            return
        except Exception as e:
            last_err = e
            print(f"Failed to download from {url}: {e}")
    raise RuntimeError(f"Could not download dataset. Last error: {last_err}")

def load_data():
    ensure_dataset()
    df = pd.read_csv(DATA_PATH)
    # Sanity: enforce columns
    expected = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    missing_cols = set(expected) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")
    return df

def compute_feature_medians(df):
    feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    return df[feats].median().to_dict()

def main():
    df = load_data()

    # Replace zeros with NaN for features where 0 is invalid/missing
    zero_as_nan = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)

    # Save medians (before scaling) for charts
    medians = compute_feature_medians(df.fillna(df.median(numeric_only=True)))

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: impute medians -> scale -> logistic regression
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear"))
    ])

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(STATS_PATH, "w") as f:
        json.dump({"medians": medians, "features": list(X.columns)}, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print("Saved model to", MODEL_PATH)
    print("Saved feature stats to", STATS_PATH)
    print("Saved metrics to", METRICS_PATH)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
