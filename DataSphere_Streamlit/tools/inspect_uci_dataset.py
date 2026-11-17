
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Load Adult dataset
adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Unique values in categorical columns
print("\nCategorical columns:")
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"{column}: {df[column].nunique()} unique values")

# Target distribution
print("\nTarget distribution:")
print(y.value_counts())
