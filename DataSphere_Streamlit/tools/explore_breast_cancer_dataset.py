
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ============================================================
# 1. LOAD THE BREAST CANCER DATASET
# ============================================================
data = load_breast_cancer()

# ============================================================
# 2. DISPLAY BASIC DATASET INFORMATION
# ============================================================
print("Dataset name: Breast Cancer Wisconsin")
print("Number of records:", data.data.shape[0])
print("Number of features:", data.data.shape[1])
print("Feature names:", data.feature_names)
print("Target names:", data.target_names)

# ============================================================
# 3. CONVERT THE DATASET INTO A DATAFRAME
# ============================================================
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add the numerical target
df["target"] = data.target

# Add readable class labels
df["diagnosis"] = [
    data.target_names[value]
    for value in data.target
]

# ============================================================
# 4. EXPLORE THE DATASET
# ============================================================
print("\nDataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nSample records:")
print(df[["mean radius", "mean texture", "target", "diagnosis"]].head())
