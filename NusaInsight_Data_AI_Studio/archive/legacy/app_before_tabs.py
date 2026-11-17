
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_diabetes,
    fetch_california_housing
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="DataSphere",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# DATASPHERE THEME
# ============================================================
st.markdown("""
<style>
    /* Main page */
    .stApp {
        background-color: #F4F7F9;
        color: #172D3A;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #173747;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DDE6EB;
        padding: 16px;
        border-radius: 12px;
    }

    /* Dataset information box */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #DDE6EB;
    }

    .stTabs [data-baseweb="tab"] {
        color: #526574;
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #087F8C;
    }

    /* Buttons */
    .stButton button {
        background-color: #087F8C;
        color: white;
        border-radius: 8px;
        border: none;
    }

    .stButton button:hover {
        background-color: #066873;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. DATASET SELECTION
# ============================================================
st.title("🌐 DataSphere")
st.subheader("Explore Data. Discover Insights. Build Better Models.")

st.write(
    "Welcome to DataSphere! Explore datasets and discover "
    "how machine learning can help us understand data."
)

st.header("Dataset Explorer")

dataset_name = st.selectbox(
    "Choose a dataset:",
    [
        "Breast Cancer",
        "Iris",
        "Diabetes",
        "California Housing"
    ]
)

# ============================================================
# 3. LOAD THE SELECTED DATASET
# ============================================================
if dataset_name == "Breast Cancer":
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

elif dataset_name == "Iris":
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

elif dataset_name == "Diabetes":
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = None

else:
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = None

# ============================================================
# 4. CREATE THE DATAFRAME
# ============================================================
df = pd.DataFrame(
    X,
    columns=feature_names
)

df["target"] = y

# Add readable labels for classification datasets
if target_names is not None:
    df["class_name"] = [
        target_names[int(value)]
        for value in y
    ]

# ============================================================
# 5. DATASET DESCRIPTION
# ============================================================
descriptions = {
    "Breast Cancer": (
        "Explore numerical measurements of cell nuclei. "
        "The target identifies samples as benign or malignant. "
        "This dataset is for educational purposes, not medical diagnosis."
    ),
    "Iris": (
        "Explore flower measurements such as sepal length, "
        "sepal width, petal length, and petal width. "
        "The target represents three iris species."
    ),
    "Diabetes": (
        "Explore numerical health-related measurements and "
        "a quantitative disease-progression target."
    ),
    "California Housing": (
        "Explore California district housing data, including "
        "median income, housing age, and population. "
        "The target is median house value in the dataset."
    )
}

st.info(descriptions[dataset_name])

# ============================================================
# 6. DISPLAY DATASET INFORMATION
# ============================================================
st.subheader(f"Dataset: {dataset_name}")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", df.shape[0])
col2.metric("Features", len(feature_names))

if target_names is not None:
    col3.metric("Classes", len(target_names))
else:
    col3.metric("Target Type", "Numerical")

# ============================================================
# 7. DISPLAY SAMPLE RECORDS
# ============================================================
st.subheader("Sample Records")

st.dataframe(
    df.head(),
    width="stretch"
)


# ============================================================
# 8. FEATURE SUMMARY
# ============================================================
st.subheader("Feature Summary")

feature_summary = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Missing Values": df.notna().sum().values
})

st.dataframe(
    feature_summary,
    width="stretch",
    hide_index=True
)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================
st.subheader("Target Distribution")

if target_names is not None:
    target_counts = df["class_name"].value_counts()

    st.bar_chart(target_counts)

else:
    st.info(
        "This dataset has a numerical target. "
        "A target distribution chart for numerical data "
        "will be added later."
    )


# ============================================================
# 10. FEATURE DISTRIBUTION
# ============================================================
st.subheader("Feature Distribution")

selected_feature = st.selectbox(
    "Choose a numerical feature:",
    feature_names
)

st.write(f"Distribution of **{selected_feature}**")

# Divide numerical values into 20 intervals
feature_bins = pd.cut(
    df[selected_feature],
    bins=20
)

# Count the records in each interval
histogram_data = feature_bins.value_counts().sort_index()

# Use readable interval labels
histogram_data.index = histogram_data.index.astype(str)

# Display the distribution chart
st.bar_chart(
    histogram_data,
    x_label="Feature value range",
    y_label="Number of records"
)



# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================
st.subheader("Correlation Heatmap")

st.caption(
    "Explore linear relationships between numerical features. "
    "Red indicates positive correlation; blue indicates negative correlation."
)

# Let users choose whether to display correlation values
show_values = st.checkbox(
    "Show correlation values",
    value=False
)

# Select only the original numerical features
correlation_data = df[feature_names]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Use a smaller figure for datasets with many features
number_of_features = len(feature_names)

if number_of_features <= 5:
    figure_size = (6, 4.5)
    label_size = 9
elif number_of_features <= 10:
    figure_size = (7, 5.5)
    label_size = 8
else:
    figure_size = (9, 7)
    label_size = 6

# Create the heatmap
fig, ax = plt.subplots(figsize=figure_size)

sns.heatmap(
    correlation_matrix,
    annot=show_values,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.75},
    annot_kws={"size": 6},
    ax=ax
)

ax.set_title(
    f"Feature Correlations — {dataset_name}",
    fontsize=11,
    pad=10
)

ax.tick_params(axis="x", labelrotation=45, labelsize=label_size)
ax.tick_params(axis="y", labelrotation=0, labelsize=label_size)

plt.tight_layout()

st.pyplot(fig, width="content")
plt.close(fig)
