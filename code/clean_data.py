import pandas as pd
df = pd.read_csv('data/insurance 2.csv')

df.columns = df.columns.str.strip()

print("COLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nCATEGORY VALUES:")
for col in ["sex", "smoker", "region"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\nNUMBER OF EXACT DUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY:")
print(df[["age", "bmi", "children", "expenses"]].describe())


print("\nROWS WITH POTENTIALLY INVALID VALUES")
invalid_rows = df[
    (df["age"] < 0) |
    (df["bmi"] <= 0) |
    (df["children"] < 0) |
    (df["expenses"] < 0)
]

print("\nALL COPIES OF DUPLICATE ROWS")
print(df[df.duplicated(keep=False)])

rows_before = len(df)
df_clean = df.drop_duplicates(keep="first")
rows_after = len(df_clean)

print(f"\nROWS BEFORE CLEANING: {rows_before}")
print(f"ROWS AFTER CLEANING: {rows_after}")
print("ROWS REMOVED:", rows_before - rows_after)
print("REMAINING DUPLICATES:", df_clean.duplicated().sum())

df_clean.to_csv("data/insurance_cleaned.csv", index=False)

print("\nCleaned data saved to 'data/insurance_cleaned.csv'.")


print(invalid_rows)

