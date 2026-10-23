
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_scoring_model.joblib"

# Flask application
app = Flask(__name__)

# Load the trained machine learning pipeline
try:
    model = joblib.load(MODEL_PATH)
    print("Credit scoring model loaded successfully!", flush=True)
except Exception as exc:
    model = None
    print(f"ERROR: Could not load model: {exc}", flush=True)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "The trained model could not be loaded. Check the models folder."
        else:
            try:
                # Collect the 20 features used by the training dataset
                applicant = {
                    "checking_status": request.form["checking_status"],
                    "duration": int(request.form["duration"]),
                    "credit_history": request.form["credit_history"],
                    "purpose": request.form["purpose"],
                    "credit_amount": int(request.form["credit_amount"]),
                    "savings_status": request.form["savings_status"],
                    "employment": request.form["employment"],
                    "installment_commitment": int(
                        request.form["installment_commitment"]
                    ),
                    "personal_status": request.form["personal_status"],
                    "other_parties": request.form["other_parties"],
                    "residence_since": int(request.form["residence_since"]),
                    "property_magnitude": request.form["property_magnitude"],
                    "age": int(request.form["age"]),
                    "other_payment_plans": request.form["other_payment_plans"],
                    "housing": request.form["housing"],
                    "existing_credits": int(request.form["existing_credits"]),
                    "job": request.form["job"],
                    "num_dependents": int(request.form["num_dependents"]),
                    "own_telephone": request.form["own_telephone"],
                    "foreign_worker": request.form["foreign_worker"],
                }

                # Create input in the same column format as training
                features = pd.DataFrame([applicant])

                # Predict category
                prediction = model.predict(features)[0]

                if str(prediction).lower() in ("1", "good"):
                    result = "Good Credit Category"
                else:
                    result = "Bad Credit Category"

                # Calculate probability of the good credit category
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = list(model.classes_)

                    good_class = next(
                        (
                            cls for cls in classes
                            if str(cls).lower() in ("1", "good")
                        ),
                        None,
                    )

                    if good_class is not None:
                        probability = round(
                            float(probabilities[classes.index(good_class)]) * 100,
                            2,
                        )

            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "Prediction failed. Please check the entered values "
                    "and confirm the form matches the trained dataset."
                )

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    print("Starting CreditScore AI web application...", flush=True)
    print("Open http://127.0.0.1:5000 in your browser.", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
