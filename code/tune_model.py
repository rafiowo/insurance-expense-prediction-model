from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold

train_df = pd.read_csv('data/insurance_train.csv')

X_train = train_df.drop(columns=["expenses"])
y_train = train_df["expenses"]

results_folder = Path("results")
models_folder = Path("models")

results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)

saved_model = joblib.load(
    models_folder / "python_random_forest_model.joblib"
)

model = clone(saved_model)

print("Training rows:", len(train_df))

parameter_grid = {
    "regressor__max_depth": [None, 10],
    "regressor__min_samples_leaf": [1, 2, 3]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error"
    },
    refit="RMSE",
    cv=cv,
    return_train_score=True,
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

print("\nStarting the parameter search")

search.fit(X_train, y_train)

raw_results = pd.DataFrame(search.cv_results_)

summary = pd.DataFrame(
    {
        "max_depth":[
            str(params["regressor__max_depth"])
            for params in raw_results["params"]
        ],
        "min_samples_leaf": [
            params["regressor__min_samples_leaf"]
            for params in raw_results["params"]
        ],
        "mean_train_RMSE": -raw_results["mean_train_RMSE"],
        "mean_validation_MAE": -raw_results["mean_test_MAE"],
        "mean_validation_RMSE": -raw_results["mean_test_RMSE"],
        "validation_RMSE_std": raw_results["std_test_RMSE"],
        "rank": raw_results["rank_test_RMSE"]
    }
)

summary = summary.sort_values(
    by="mean_validation_RMSE"
)

print("\nTUNING RESULTS:")
print(summary.round(2).to_string(index=False))

print("\nSELECTED SETTINGS:")
print(search.best_params_)

print(
    "\nSelected mean validation RMSE",
    round(-search.best_score_, 2)
)

summary.to_csv(
    results_folder / "python_random_forest_tuning.csv",
    index=False
)

tuned_model_path = models_folder / "python_random_forest_tuned.joblib"

joblib.dump(
    search.best_estimator_,
    models_folder / "python_random_forest_tuned.joblib",
    compress = 3
)

print("\nSaved tuning results.")
print("Saved models/python_random_forest_tuned.joblib")

