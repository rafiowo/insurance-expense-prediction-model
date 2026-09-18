from pathlib import Path
import math
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


train_df = pd.read_csv('data/insurance_train.csv')

X_train = train_df.drop(columns=["expenses"])
y_train = train_df["expenses"]

results_folder = Path("results")
models_folder = Path("models")

results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)

print("Training rows: ", len(train_df))
print("Input shapoe: ", X_train.columns.tolist())


numerical_columns = ["age", "bmi", "children"]
categorical_columns = ["sex", "smoker", "region"]

numerical_processor = SimpleImputer(strategy="mean")

categorical_preprocessor = Pipeline(
    steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_processor, numerical_columns),
        ("categorical", categorical_preprocessor, categorical_columns)
    ]
)

forest = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None, min_samples_leaf=1, max_features=1.0, n_jobs=-1)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", forest)
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning five-fold cross-validation...")

cv_predictions = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)


mae = mean_absolute_error(y_train, cv_predictions)
mse = mean_squared_error(y_train, cv_predictions)
rmse = math.sqrt(mse)

print("\nCROSS-VALIDATION RESULTS:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

metrics = pd.DataFrame(
    [{
        "model": "Random Forest Regressor",
        "MAE": mae,
        "RMSE": rmse,
        "trees": 200,
        "cv_seed": 42,
        "model_seed": 42

    }]
)

metrics.to_csv(results_folder / "01_model_metrics.csv", index=False)

prediction_table = pd.DataFrame({
    "training_row": X_train.index,
    "actual_expenses": y_train.to_numpy(),
    "predicted_expenses": cv_predictions
})

prediction_table.to_csv(results_folder / "02_cv_predictions.csv", index=False)

print("\nTraining the pipeline on all training rows...")
model.fit(X_train, y_train)

model_path = models_folder / "python_random_forest_model.joblib"
joblib.dump(model, model_path, compress=3)

print("Saved trained pipeline:", model_path)
print("Saved validation results and predictions in results folder.")
print("Input columns:", X_train.columns.tolist())
print("Input shape", X_train.shape)

