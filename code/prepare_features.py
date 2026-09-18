import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('data/insurance_train.csv')

X_train = df.drop(columns=["expenses"])
y_train = df["expenses"]

print("INPUT COLUMNS:")
print(X_train.columns.tolist())

print("\nINPUT SHAPE:")
print(X_train.shape)

print("\nTARGET SHAPE:")
print(y_train.shape)

print("\nTARGET SHAPE:")
print(y_train.shape)

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

print("\nPreprocessing configuration created successfully.")
print("\nNumerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

