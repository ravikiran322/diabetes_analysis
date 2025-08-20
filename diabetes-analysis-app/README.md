# Diabetes Analysis App (Flask + ML)

A complete, ready-to-run Diabetes Analysis web app using the PIMA Indians Diabetes dataset.
It trains a machine learning model (Logistic Regression in a scikit-learn Pipeline) and serves
a simple Flask web UI to take inputs, predict risk, show probability, and visualize how the
user compares to dataset medians.

## ✨ Features
- Trainable ML model with proper preprocessing (median imputation, standardization).
- Balanced Logistic Regression for class imbalance.
- Save & load model with joblib.
- Frontend with Bootstrap + Chart.js (bar chart + radial gauge).
- Simple CSV logging of predictions.
- Clean, minimal UI.

## 🧱 Project Structure
```
diabetes-analysis-app/
│── data/
│   └── diabetes.csv              # auto-downloaded on first training run if missing
│── model/
│   ├── diabetes_model.pkl        # trained ML pipeline
│   ├── feature_stats.json        # medians for charts
│   └── metrics.json              # basic training metrics
│── static/
│   └── style.css                 # styles
│── templates/
│   ├── index.html                # input form
│   └── result.html               # prediction + charts
│── app.py                        # Flask server
│── train_model.py                # training script
│── requirements.txt
│── README.md
```

## 🚀 Quickstart
1. **Create & activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (downloads dataset if missing):**
   ```bash
   python train_model.py
   ```

4. **Run the web app:**
   ```bash
   python app.py
   ```
   Open the printed URL in your browser (usually http://127.0.0.1:5000).

## 🧪 Test Inputs
Try realistic values (non-zero for Glucose, BloodPressure, BMI). For example:
- Pregnancies: 2
- Glucose: 130
- BloodPressure: 72
- SkinThickness: 25
- Insulin: 80
- BMI: 30.5
- DiabetesPedigreeFunction: 0.5
- Age: 35

## 📌 Notes
- Some features in the PIMA dataset use `0` to represent missing values (notably Glucose, BloodPressure, SkinThickness, Insulin, BMI). The training script replaces those with `NaN` and imputes with median.
- Metrics are saved to `model/metrics.json` after training.

## 📄 License
MIT (for this sample project). Dataset is from the public PIMA Indians Diabetes dataset (several mirrors exist online).
