from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


results_folder = Path("results")

df = pd.read_csv(
    results_folder / "python_test_predictions.csv"
)

actual = df["expenses"]
predicted = df["predicted_expenses"]

residuals = actual - predicted

df["residuals"] = residuals
df["absolute_error"] = residuals.abs()

print("Test records:", len(df))

lower = min(actual.min(), predicted.min())
upper = max(actual.max(), predicted.max())

padding = (upper - lower) * 0.05

lower = lower - padding
upper = upper - padding

plt.figure(figsize=(7, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.6,
    s=25,
    color="steelblue"
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    color="black",
    label="Perfect prediciton"
)

plt.xlim(lower, upper)
plt.xlim(lower, upper)

plt.title("Actual vs Predicted Medical Expenses")
plt.xlabel("Actual expenses")
plt.ylabel("Predicted expenses")
plt.legend()

plt.tight_layout()

plt.savefig(
    results_folder / "05_actual_vs_predicted.png",
    dpi=200
)

plt.close()

plt.figure(figsize=(8, 5))

plt.scatter(
    predicted,
    residuals,
    alpha=0.6,
    s=25,
    color="darkorange"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    label="Zero error"
)

plt.title("Residuals vs Predicted Medical Expenses")
plt.xlabel("Predicted expenses")
plt.ylabel("Residuals: actual minus predicted")
plt.legend()

plt.tight_layout()

plt.savefig(
    results_folder / "06_residuals_vs_predicted,png",
    dpi=200
)

plt.close()

plt.figure(figsize=(8, 5))

plt.hist(
    residuals,
    bins=25,
    color="seagreen",
    edgecolor="white"
)

plt.axvline(
    x=0,
    color="black",
    linestyle="--",
    label="Zero error"
)

plt.title("Distribution of predicted residuals")
plt.xlabel("Residuals: actual minus predicted")
plt.ylabel("Number of test records")
plt.legend()

plt.tight_layout()

plt.savefig(
    results_folder  / "07_residual_distribution.png",
    dpi = 200
)

plt.close()

underpredictions = (residuals > 0).sum()
overpredictions = (residuals < 0 ).sum()
exact_matches = (residuals == 0).sum()

print("\nERROR SUMMARY")
print(f"Mean residual: {residuals.mean():.2f}")
print(f"Median residual: {residuals.median():.2f}")

print(
    f"Median absolute error: "
    f"{df['absolute_error'].median():.2f}"
)

print(
    f"Largest absolute error: "
    f"{df['absolute_error'].max():.2f}"
)

print("Underpredictions:", underpredictions)
print("Overpredictions", overpredictions)
print("Exact matches", exact_matches)

largest_errors = df.sort_values(
    by="absolute_error",
    ascending=False
).head(10)

print("\nTEN LARGEST ERRORS")

print(
    largest_errors[
        [
            "age",
            "bmi",
            "smoker",
            "expenses",
            "predicted_expenses",
            "residuals",
            "absolute_error"
        ]
    ].round(2).to_string(index=False)
)

largest_errors.to_csv(
    results_folder / "python_test_largest_errors.csv",
    index=False
)

print("\nSaved error-analysis graphs and largest-error table.")

