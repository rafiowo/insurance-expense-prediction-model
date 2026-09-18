from pathlib import Path
import math
import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

test_df = pd.read_csv("data/insurance_test.csv")

X_test = test_df.drop(columns=["expenses"])
y_test = test_df["expenses"]

results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

print("Test rows:", len(test_df))
print("Test input columns:", X_test.columns.tolist())

model_path = Path(
    "models/python_random_forest_tuned.joblib"
)

model = joblib.load(model_path)

print("\nLoaded model:", model_path)

predictions = model.predict(X_test)

print("Predictions generated:", len(predictions))

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nPYTHON RANDOM FOREST - TEST RESULTS")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}" )


train_df = pd.read_csv("data/insurance_train.csv")

training_mean = train_df["expenses"].mean()

baseline_predictions = [
    training_mean
] * len(test_df)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = math.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

baseline_r2 = r2_score(
    y_test, baseline_predictions
)

print("\nTRAINING-MEAN BASELINE - TEST RESULTS")
print(f"Constant prediction: {training_mean:.2f}")
print(f"MAE: {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R2: {baseline_r2:.4f}")

metrics = pd.DataFrame(
    [
        {
            "model": "Training-mean baseline",
            "evaluation": "Held-out test set",
            "rows": len(test_df),
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "R2": baseline_r2
        },
        {
            "model": "Python tuned Random Forest",
            "evaluation": "Held-out test set",
            "rows": len(test_df),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    ]
)

metrics.to_csv(
    results_folder / "python_test_metrics.csv",
    index=False
)

prediction_table = test_df.copy()

prediction_table["predicted_expenses"] = predictions

prediction_table["residual"] = (
    prediction_table["expenses"]
    - prediction_table["predicted_expenses"]
)

prediction_table["absolute_error"] = (
    prediction_table["residual"].abs()
)

prediction_table.to_csv(
    results_folder / "python_test_predictions.csv",
    index=False
)

print("\nFIRST FIVE PREDICTIONS:")
print(
    prediction_table[
        [
            "expenses",
            "predicted_expenses",
            "residual",
            "absolute_error"
        ]
    ].head().round(2)
)

print("\nSaved test metrics and predictions in results.")