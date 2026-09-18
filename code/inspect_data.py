import pandas as pd

df = pd.read_csv('data/insurance 2.csv')

print("FIRST 5 ROWS")
print(df.head())

print("\nROWS AND COLUMNS")
print(df.shape)

print("\nCOLUMNS NAMES")
print(df.columns.tolist())

print("\nDATA TYPES AND NON-MISSING COUNTS")
df.info()

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nEXACT DUPLICATE ROWS")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY")
print(df.describe())

