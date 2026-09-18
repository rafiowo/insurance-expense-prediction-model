import argparse
from pathlib import Path

import joblib
import pandas as pd


parser = argparse.ArgumentParser(
    description="Predict medical expenses using a saved model."
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to a CSV containing the input features."
)

parser.add_argument(
    "--output",
    required=True,
    help="Path where predictions will be saved."
)

parser.add_argument(
    "--model",
    default="models/python_random_forest_tuned.joblib",
    help="Path to the saved prediction pipeline."
)

args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output)
model_path = Path(args.model)

# Avoid accidentally overwriting the input or saved model.
if output_path.resolve() in {
    input_path.resolve(),
    model_path.resolve()
}:
    parser.error(
        "The output path must differ from the input and model paths."
    )

required_columns = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region"
]

input_df = pd.read_csv(input_path)

missing_columns = [
    column
    for column in required_columns
    if column not in input_df.columns
]

if missing_columns:
    parser.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

if input_df.empty:
    parser.error("The input CSV contains no data rows.")

# Select only model inputs, in a consistent order.
X_new = input_df[required_columns].copy()

# Check that numerical inputs contain numbers or missing values.
for column in ["age", "bmi", "children"]:
    try:
        X_new[column] = pd.to_numeric(
            X_new[column],
            errors="raise"
        )
    except (ValueError, TypeError):
        parser.error(
            f"Column '{column}' must contain numbers "
            "or blank values."
        )

model = joblib.load(model_path)

predictions = model.predict(X_new)

output_df = input_df.copy()
output_df["predicted_expenses"] = predictions

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_df.to_csv(output_path, index=False)

print("Input records:", len(input_df))
print("Predictions generated:", len(predictions))
print("Saved predictions to:", output_path)