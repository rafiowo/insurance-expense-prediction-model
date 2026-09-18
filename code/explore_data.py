from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/insurance_train.csv')

output_folder = Path("results")
output_folder.mkdir(parents=True, exist_ok=True)

print("training rows: ", len(df))

#Medical Expenses Distribution Histogram

plt.figure(figsize=(8, 5))

plt.hist(df["expenses"], bins=30, color='steelblue', edgecolor='white')

plt.title("Distribution of Medical Expenses - Training Set")
plt.xlabel("Medical Expenses")
plt.ylabel("Number of Records")
plt.show()

plt.tight_layout()
plt.savefig(output_folder / "01_expenses_distribution.png", dpi=200)
plt.close()

#Age vs Expenses Scatter Plot

plt.figure(figsize=(8, 5))
plt.scatter(df["age"], df["expenses"], alpha=0.2, s=20)

plt.title("Age vs Medical Expenses - Training Set")
plt.xlabel("Age")
plt.ylabel("Medical Expenses")
plt.show()

plt.tight_layout()
plt.savefig(output_folder / "02_age_vs_expenses.png", dpi=200)
plt.close()

#BMI vs Expenses Scatter Plot

plt.figure(figsize=(8, 5))
plt.scatter(df["bmi"], df["expenses"], alpha=0.2, s=20, color='darkorange')

plt.title("BMI vs Medical Expenses - Training Set")
plt.xlabel("BMI")
plt.ylabel("Medical Expenses")
plt.show()

plt.tight_layout()
plt.savefig(output_folder / "03_bmi_vs_expenses.png", dpi=200)
plt.close()

#Expenses by Smoking status Box Plot

non_smokers = df.loc[df["smoker"] == "no", "expenses"]
smoker_expenses = df.loc[df["smoker"] == "yes", "expenses"]

plt.figure(figsize=(8, 5))
plt.boxplot([non_smokers, smoker_expenses])

plt.xticks([1, 2], ["Non-Smokers", "Smokers"])

plt.title("Medical Expenses by Smoking Status - Training Set")
plt.xlabel("Smoking Status")
plt.ylabel("Medical Expenses")

plt.tight_layout()
plt.savefig(output_folder / "04_expenses_by_smoking_status.png", dpi=200)
plt.close()

#Numercal Summary Statistics

print("\nEXPENESES BY SMOKING STATUS")

smoking_summary = (df.groupby("smoker")["expenses"]
                   .agg(["count", "mean", "median"])
)

print(smoking_summary)

smoking_summary.to_csv(output_folder / "05_expenses_by_smoking_status.csv", index=True)

print("\nFinished. Graphs and summary saved in results.")


