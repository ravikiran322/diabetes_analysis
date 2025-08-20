from flask import Flask, render_template, request
import os
import json
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "diabetes_model.pkl")
STATS_PATH = os.path.join("model", "feature_stats.json")
LOG_PATH = os.path.join("data", "predictions_log.csv")

# Load model and stats at startup
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
with open(STATS_PATH, "r") as f:
    stats = json.load(f)
feature_names = stats["features"]
medians = [stats["medians"][k] for k in feature_names]

def get_recommendations(prob, user_input):
    tips = []
    # Simple rule-based hints (non-medical)
    if user_input.get("BMI", 0) >= 25:
        tips.append("Aim for a balanced diet and regular physical activity to manage BMI.")
    if user_input.get("Glucose", 0) >= 140:
        tips.append("Your glucose is on the higher side—consider reducing refined sugar and monitor regularly.")
    if user_input.get("BloodPressure", 0) >= 130:
        tips.append("Watch your salt intake and consider BP monitoring.")
    if user_input.get("Age", 0) >= 45:
        tips.append("Age is a risk factor; schedule periodic checkups.")
    if user_input.get("DiabetesPedigreeFunction", 0) >= 0.8:
        tips.append("Family history indicator is elevated; be proactive with lifestyle changes.")
    if not tips:
        tips.append("Maintain a balanced diet (fiber-rich foods, lean proteins).")
        tips.append("Stay active (150+ minutes of moderate exercise per week).")
    if prob >= 0.5:
        tips.insert(0, "Result suggests higher risk. Please consult a healthcare professional for proper testing.")
    return tips

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not found. Please run 'python train_model.py' first.", 500

    # Collect and order inputs
    values = []
    user_input = {}
    for name in feature_names:
        raw = request.form.get(name)
        try:
            val = float(raw)
        except:
            val = 0.0
        user_input[name] = val
        values.append(val)

    # Create DataFrame with expected columns
    X = pd.DataFrame([values], columns=feature_names)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    # Log to CSV
    os.makedirs("data", exist_ok=True)
    log_row = user_input.copy()
    log_row.update({"probability": prob, "prediction": pred})
    pd.DataFrame([log_row]).to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

    recs = get_recommendations(prob, user_input)

    return render_template(
        "result.html",
        probability=prob,
        predicted_class=pred,
        feature_names=feature_names,
        user_values=[user_input[k] for k in feature_names],
        medians=medians,
        recommendations=recs
    )

if __name__ == "__main__":
    # Helpful message if model isn't trained yet
    if not os.path.exists(MODEL_PATH):
        print("❗ Model not found. Run: python train_model.py")
    app.run(host="127.0.0.1", port=5000, debug=True)
