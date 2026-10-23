
from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "credit_dataset.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Predict creditworthiness using the trained model."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a CSV containing applicant feature columns.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train"
        )

    model = joblib.load(MODEL_PATH)

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        applicants = pd.read_csv(input_path)
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "Dataset not found. Run: python -m src.train"
            )
        dataset = pd.read_csv(DATA_PATH)
        applicants = dataset.drop(columns=["creditworthy"]).head(1)
        print("Demo mode: predicting the first dataset record.")
        print("For a new applicant, use --input with your own CSV.")

    if applicants.empty:
        raise ValueError("The input CSV contains no applicant records.")

    expected_columns = list(model.feature_names_in_)
    missing = [c for c in expected_columns if c not in applicants.columns]

    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    applicants = applicants[expected_columns]
    predictions = model.predict(applicants)
    probabilities = model.predict_proba(applicants)[:, 1]

    for i, (prediction, probability) in enumerate(
        zip(predictions, probabilities), start=1
    ):
        category = (
            "Good credit category"
            if prediction == 1
            else "Bad credit category"
        )
        print(f"\nApplicant {i}")
        print(f"Predicted category: {category}")
        print(f"Estimated probability of good category: {probability:.1%}")

    print(
        "\nDisclaimer: Educational model output only. "
        "Not an official CIBIL score or lending decision."
    )


if __name__ == "__main__":
    main()
