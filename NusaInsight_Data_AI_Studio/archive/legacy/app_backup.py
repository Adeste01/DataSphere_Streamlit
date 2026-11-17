
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Page configuration
st.set_page_config(
    page_title="DataSphere",
    page_icon="📊",
    layout="wide"
)

# Load the dataset
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target
df["diagnosis"] = [
    data.target_names[value]
    for value in data.target
]

# Page title and introduction
st.title("🌐 DataSphere")
st.subheader("Explore Data. Discover Insights. Build Better Models.")

st.write(
    "Welcome to DataSphere! Explore datasets and discover "
    "how machine learning can help us understand data."
)

# Dataset overview
st.header("Dataset Explorer")
st.write("Dataset: Breast Cancer Wisconsin")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", df.shape[0])
col2.metric("Features", len(data.feature_names))
col3.metric("Classes", len(data.target_names))

# Display the dataset
st.subheader("Sample Records")
st.dataframe(df.head(), use_container_width=True)
