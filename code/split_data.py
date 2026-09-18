import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/insurance_cleaned.csv')

if df.duplicated().sum() > 0:
    raise ValueError("The cleaned dataset still contains duplicate rows.")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


assert len(train_df) + len(test_df) == len(df)
assert set(train_df.index).isdisjoint(test_df.index)

train_df.to_csv("data/insurance_train.csv", index=False)
test_df.to_csv("data/insurance_test.csv", index=False)

print("TOTAL ROWS:", len(df))
print("TRAINING ROWS:", len(train_df))
print("TEST ROWS:", len(test_df))

print("\nSaved data/train.csv")
print("\nSaved data/test.csv")