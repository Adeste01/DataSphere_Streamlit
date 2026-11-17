import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_diabetes,
    fetch_california_housing,
)
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="DataSphere | Dataset Explorer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. DATASPHERE THEME
# ============================================================
st.markdown(
    """
    <style>
        :root {
            --ds-navy: #102632;
            --ds-teal: #087F8C;
            --ds-teal-light: #E4F3F3;
            --ds-page: #F4F7F9;
            --ds-border: #DDE6EB;
            --ds-text: #203744;
            --ds-muted: #647783;
        }
        .stApp {
            background-color: var(--ds-page);
            color: var(--ds-text);
        }
        .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3 {
            color: #173747;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102632 0%, #163746 100%);
            border-right: 1px solid #244653;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 8px;
            padding: 0.25rem 0.45rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: #F4F7F9;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #D6E3E8;
        }
        /* Regression-only metric sizing: prevent long values (especially MSE) from truncating */
        .st-key-regression_metrics [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            white-space: nowrap;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid var(--ds-border);
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 112px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--ds-muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--ds-navy);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        .ds-eyebrow {
            color: var(--ds-teal);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .ds-subtitle {
            color: var(--ds-muted);
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1.25rem;
        }
        .ds-card {
            background: #FFFFFF;
            border: 1px solid var(--ds-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            height: 100%;
            box-shadow: 0 2px 8px rgba(16,38,50,0.035);
        }
        .ds-section-label {
            color: #087F8C;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stButton"]) {
            margin-top: 0.25rem;
        }
        .ds-card-title {
            color: var(--ds-navy);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .ds-card-text {
            color: var(--ds-muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .stButton button {
            background-color: var(--ds-teal);
            color: white;
            border: 1px solid var(--ds-teal);
            border-radius: 9px;
        }
        .stButton button:hover {
            background-color: #066873;
            color: white;
            border-color: #066873;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--ds-border);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 12px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--ds-border);
            border-radius: 10px;
            overflow: hidden;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. DATASET LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(dataset_name):
    """Load a supported dataset and return its DataFrame and metadata."""

    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df["class_name"] = [target_names[int(value)] for value in y]

    elif dataset_name == "Iris":
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df["class_name"] = [target_names[int(value)] for value in y]

    elif dataset_name == "Diabetes":
        data = load_diabetes()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = None

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

    elif dataset_name == "California Housing":
        data = fetch_california_housing()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = None

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

    elif dataset_name == "Wine (UCI)":
        # UCI Wine Recognition dataset, ID 109
        wine = fetch_ucirepo(id=109)
        X = wine.data.features.copy()
        y = wine.data.targets.copy()

        # UCI target is a cultivar/class label (usually 1, 2, or 3).
        target_column = y.columns[0]
        y_values = y[target_column]

        feature_names = X.columns.tolist()
        target_names = sorted(y_values.dropna().unique().tolist())

        df = X.copy()
        df["target"] = y_values.values
        df["class_name"] = y_values.astype(str).values

        data = wine

    elif dataset_name == "Wine Quality (UCI)":
        # UCI Wine Quality dataset, ID 186
        quality = fetch_ucirepo(id=186)
        X = quality.data.features.copy()
        y = quality.data.targets.copy()
        target_column = y.columns[0]
        y_values = y[target_column]

        feature_names = X.columns.tolist()
        target_names = None
        df = X.copy()
        df["target"] = y_values.values
        data = quality

    elif dataset_name == "Heart Disease (UCI)":
        # UCI Heart Disease dataset, ID 45 (Cleveland subset)
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        target_column = y.columns[0]
        y_values = y[target_column]

        feature_names = X.columns.tolist()
        # Heart Disease target is numerical (0 = no disease; >0 = disease level).
        df = X.copy()
        df["target"] = y_values.values
        target_names = ["No disease", "Heart disease"]
        df["class_name"] = y_values.apply(
            lambda value: "No disease" if pd.notna(value) and float(value) == 0
            else ("Heart disease" if pd.notna(value) else "Missing")
        ).values
        data = heart

    elif dataset_name == "Bank Marketing (UCI)":
        # UCI Bank Marketing dataset, ID 222
        bank = fetch_ucirepo(id=222)
        X = bank.data.features.copy()
        y = bank.data.targets.copy()
        target_column = y.columns[0]
        y_values = y[target_column]

        feature_names = X.columns.tolist()
        target_names = sorted(y_values.dropna().astype(str).unique().tolist())
        df = X.copy()
        df["target"] = y_values.values
        df["class_name"] = y_values.astype(str).values
        data = bank

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return df, feature_names, target_names


DESCRIPTIONS = {
    "Breast Cancer": (
        "Explore numerical measurements of cell nuclei. The target identifies "
        "samples as benign or malignant. This dataset is for educational "
        "purposes and must not be used for medical diagnosis."
    ),
    "Iris": (
        "Explore flower measurements such as sepal length, sepal width, petal "
        "length, and petal width. The target represents three iris species."
    ),
    "Diabetes": (
        "Explore numerical health-related measurements and a quantitative "
        "target representing disease progression."
    ),
    "California Housing": (
        "Explore California district housing data, including median income, "
        "housing age, rooms, and population. The target is the dataset's "
        "median house value measure."
    ),
    "Wine (UCI)": (
        "Explore 13 numerical chemical measurements of wines from three "
        "cultivars. Features include alcohol, magnesium, flavanoids, and "
        "color intensity. This dataset is suitable for classification "
        "and clustering."
    ),
    "Wine Quality (UCI)": (
        "Explore physicochemical laboratory measurements of red and white "
        "wine. The numerical target is a quality score, making this dataset "
        "suitable for regression or for creating quality categories."
    ),
    "Heart Disease (UCI)": (
        "Explore clinical measurements from the UCI Heart Disease dataset. "
        "The target indicates absence or presence/severity of heart disease. "
        "This dataset is for educational analysis, not medical diagnosis."
    ),
    "Bank Marketing (UCI)": (
        "Explore customer and campaign information from a Portuguese bank's "
        "marketing campaign. The target indicates whether a client subscribed "
        "to a term deposit. Several input features are categorical."
    ),
}

DATASET_TASKS = {
    "Breast Cancer": "Classification",
    "Iris": "Classification / Clustering",
    "Diabetes": "Regression",
    "California Housing": "Regression",
    "Wine (UCI)": "Classification / Clustering",
    "Wine Quality (UCI)": "Regression / Classification",
    "Heart Disease (UCI)": "Classification",
    "Bank Marketing (UCI)": "Classification",
}

CLASSIFICATION_DATASETS = [
    "Breast Cancer", "Iris", "Wine (UCI)", "Heart Disease (UCI)", "Bank Marketing (UCI)"
]

# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🌐 DataSphere")
    st.caption("Explore Data. Discover Insights. Build Better Models.")
    st.divider()
    selected_page = st.radio(
        "Navigation",
        ["Home", "Dataset Explorer", "Machine Learning Lab"],
        index=1,
        key="datasphere_navigation",
    )
    st.divider()
    st.caption("DataSphere • Data Science learning platform")

# ============================================================

if selected_page == "Dataset Explorer":
    # 5. PAGE HEADER AND DATASET SELECTOR
    # ============================================================
    st.markdown(
        '<div class="ds-eyebrow">DATASPHERE • DATA UNDERSTANDING</div>',
        unsafe_allow_html=True,
    )
    st.title("Dataset Explorer")
    st.markdown(
        '<div class="ds-subtitle">Explore available datasets, inspect their quality, '
        'and discover patterns before training a machine-learning model.</div>',
        unsafe_allow_html=True,
    )

    selector_col, info_col = st.columns([1.65, 1], gap="large")

    with selector_col:
        dataset_name = st.selectbox(
            "Select a dataset",
            [
                "Breast Cancer", "Iris", "Diabetes", "California Housing", "Wine (UCI)",
                "Wine Quality (UCI)", "Heart Disease (UCI)", "Bank Marketing (UCI)"
            ],
            key="dataset_selector",
        )

    with info_col:
        target_label = "Class label" if dataset_name in CLASSIFICATION_DATASETS else "Numerical value"
        st.markdown(
            f"""
            <div class="ds-card">
                <div class="ds-card-title">Selected dataset</div>
                <div class="ds-card-text"><b>Source:</b> {"UCI Machine Learning Repository" if "(UCI)" in dataset_name else "scikit-learn"}</div>
                <div class="ds-card-text"><b>Typical task:</b> {DATASET_TASKS[dataset_name]}</div>
                <div class="ds-card-text"><b>Target:</b> {target_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner(f"Loading {dataset_name} dataset..."):
        try:
            df, feature_names, target_names = load_dataset(dataset_name)
        except Exception as error:
            st.error(
                "The dataset could not be loaded. Check your internet connection "
                "and the ucimlrepo installation, then try again."
            )
            st.exception(error)
            st.stop()

    st.info(DESCRIPTIONS[dataset_name])

    # ============================================================
    # 6. DATASET OVERVIEW METRICS
    # ============================================================
    st.subheader(f"{dataset_name} overview")

    total_missing = int(df.isna().sum().sum())
    complete_records = int(df.notna().all(axis=1).sum())

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total records", f"{len(df):,}")
    metric2.metric("Features", len(feature_names))
    metric3.metric(
        "Classes" if target_names is not None else "Target type",
        len(target_names) if target_names is not None else "Numerical",
    )
    metric4.metric("Missing values", f"{total_missing:,}")

    # ============================================================
    # 7. EXPLORER TABS
    # ============================================================
    tab_overview, tab_sample, tab_distribution, tab_correlation, tab_quality = st.tabs(
        [
            "📊 Overview",
            "📋 Sample Data",
            "📈 Feature Distributions",
            "🔗 Correlation",
            "🔍 Data Quality",
        ]
    )

    # ============================================================
    # TAB 1: OVERVIEW
    # ============================================================
    with tab_overview:
        st.markdown("### About this dataset")
        st.markdown(
            f'<div class="ds-card ds-card-text">{DESCRIPTIONS[dataset_name]}</div>',
            unsafe_allow_html=True,
        )

        left_chart, right_info = st.columns([1.35, 1], gap="large")

        with left_chart:
            st.markdown("#### Target distribution")

            if target_names is not None:
                if dataset_name in ["Wine (UCI)", "Bank Marketing (UCI)", "Heart Disease (UCI)"]:
                    target_counts = df["class_name"].value_counts().sort_index()
                else:
                    target_counts = (
                        df["class_name"]
                        .value_counts()
                        .reindex(list(target_names), fill_value=0)
                    )

                fig_target, ax_target = plt.subplots(figsize=(6.2, 3.5))
                sns.barplot(
                    x=target_counts.index,
                    y=target_counts.values,
                    color="#087F8C",
                    ax=ax_target,
                )
                ax_target.set_xlabel("Class")
                ax_target.set_ylabel("Number of records")
                ax_target.set_title("Records per class", pad=10)
                ax_target.tick_params(axis="x", labelrotation=0)
                ax_target.grid(axis="y", alpha=0.2)
                ax_target.set_axisbelow(True)
                fig_target.tight_layout()
                st.pyplot(fig_target, width="stretch")
                plt.close(fig_target)

                st.dataframe(
                    target_counts.rename_axis("Class").reset_index(name="Records"),
                    width="stretch",
                    hide_index=True,
                )
            else:
                fig_target, ax_target = plt.subplots(figsize=(6.2, 3.5))
                sns.histplot(
                    df["target"],
                    bins=20,
                    color="#087F8C",
                    edgecolor="white",
                    ax=ax_target,
                )
                ax_target.set_title("Distribution of target values", pad=10)
                ax_target.set_xlabel("Target value")
                ax_target.set_ylabel("Number of records")
                ax_target.grid(axis="y", alpha=0.2)
                ax_target.set_axisbelow(True)
                fig_target.tight_layout()
                st.pyplot(fig_target, width="stretch")
                plt.close(fig_target)

                st.caption(
                    "This histogram groups numerical target values into intervals "
                    "to show where records are concentrated."
                )

        with right_info:
            st.markdown("#### Dataset summary")
            summary_items = [
                ("Dataset", dataset_name),
                ("Source", "UCI Machine Learning Repository" if "(UCI)" in dataset_name else "scikit-learn"),
                ("Number of features", f"{len(feature_names):,}"),
                ("Number of records", f"{len(df):,}"),
                ("Missing values", f"{total_missing:,}"),
                ("Suggested task", DATASET_TASKS[dataset_name]),
            ]

            summary_df = pd.DataFrame(summary_items, columns=["Item", "Details"])
            st.dataframe(summary_df, width="stretch", hide_index=True)

            st.markdown("#### Quick interpretation")
            if target_names is not None:
                interpretation = (
                    "This dataset contains labelled classes and is suitable for "
                    "classification: a model learns to predict a class from input features."
                )
                if dataset_name in ["Iris", "Wine (UCI)"]:
                    interpretation += (
                        " It can also be explored with clustering to discover groups "
                        "based on feature similarity without using class labels."
                    )
            else:
                interpretation = (
                    "This dataset has a numerical target and is suitable for regression, "
                    "where a model predicts a numerical value."
                )

            st.markdown(
                f'<div class="ds-card ds-card-text">{interpretation}</div>',
                unsafe_allow_html=True,
            )

    # ============================================================
    # TAB 2: SAMPLE DATA
    # ============================================================
    with tab_sample:
        st.markdown("### Sample records")
        st.caption(
            "Preview dataset records, review feature-level information, "
            "or download the complete dataset for further analysis."
        )

        # Preview controls and download action
        preview_col, download_col = st.columns([1.2, 1], gap="large")

        with preview_col:
            rows_to_show = st.selectbox(
                "Number of records to preview",
                [5, 10, 20, 50],
                index=0,
                key="preview_rows",
            )

        csv_data = df.to_csv(index=False).encode("utf-8")
        with download_col:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Download selected dataset as CSV",
                data=csv_data,
                file_name=f"datasphere_{dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        preview_count = min(rows_to_show, len(df))
        st.markdown(
            f'<div class="ds-section-label">DATA PREVIEW • {preview_count:,} OF {len(df):,} RECORDS</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df.head(rows_to_show),
            width="stretch",
            hide_index=True,
            height=320,
        )

        st.divider()
        st.markdown("### Feature summary")
        st.caption(
            "A quick overview of each column, its data type, and the number "
            "of available or missing values."
        )

        feature_summary = pd.DataFrame(
            {
                "Feature": df.columns,
                "Data type": df.dtypes.astype(str).values,
                "Non-missing values": df.notna().sum().values,
                "Missing values": df.isna().sum().values,
            }
        )

        st.dataframe(
            feature_summary,
            width="stretch",
            hide_index=True,
            height=360,
        )

    # ============================================================
    # TAB 3: FEATURE DISTRIBUTIONS
    # ============================================================
    with tab_distribution:
        st.subheader("Feature distributions")
        st.caption(
            "A histogram groups numerical values into intervals. Taller bars mean "
            "more records fall within that interval."
        )

        numeric_features = [
            column for column in feature_names
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_features = [
            column for column in feature_names
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        
        feature_type = st.radio(
            "Feature type",
            ["Numerical", "Categorical"],
            horizontal=True,
            key="distribution_feature_type",
        )

        available_features = (
            numeric_features
            if feature_type == "Numerical"
            else categorical_features
        )

        if not available_features:
            st.info(
                f"No {feature_type.lower()} features are available "
                "in this dataset."
            )

        else:
            selected_feature = st.selectbox(
                "Choose a feature",
                available_features,
                key="distribution_feature",
            )

            feature_values = df[selected_feature].dropna()

            if feature_type == "Numerical":

                stat1, stat2, stat3, stat4 = st.columns(4)

                stat1.metric("Mean", f"{feature_values.mean():.3f}")
                stat2.metric("Median", f"{feature_values.median():.3f}")
                stat3.metric("Minimum", f"{feature_values.min():.3f}")
                stat4.metric("Maximum", f"{feature_values.max():.3f}")

                # Balance the histogram with a compact interpretation panel.
                chart_col, insight_col = st.columns([2.15, 1], gap="large")

                with chart_col:
                    fig_dist, ax_dist = plt.subplots(figsize=(8, 4.2))

                    sns.histplot(
                        feature_values,
                        bins=20,
                        kde=False,
                        color="#087F8C",
                        edgecolor="white",
                        alpha=0.88,
                        ax=ax_dist,
                    )

                    ax_dist.axvline(
                        feature_values.mean(),
                        color="#E09F3E",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean: {feature_values.mean():.2f}",
                    )
                    ax_dist.axvline(
                        feature_values.median(),
                        color="#C43D4B",
                        linestyle=":",
                        linewidth=2,
                        label=f"Median: {feature_values.median():.2f}",
                    )

                    ax_dist.set_title(
                        f"Distribution of {selected_feature}",
                        fontsize=12,
                        fontweight="bold",
                        pad=10,
                    )
                    ax_dist.set_xlabel(selected_feature, fontsize=9)
                    ax_dist.set_ylabel("Number of records", fontsize=9)
                    ax_dist.tick_params(axis="both", labelsize=8)
                    ax_dist.grid(axis="y", alpha=0.22)
                    ax_dist.set_axisbelow(True)
                    ax_dist.spines["top"].set_visible(False)
                    ax_dist.spines["right"].set_visible(False)
                    ax_dist.legend(frameon=False, loc="upper right", fontsize=8)
                    fig_dist.tight_layout()
                    st.pyplot(fig_dist, width="stretch")
                    plt.close(fig_dist)

                with insight_col:
                    st.markdown("#### Distribution insights")
                    mean_value = feature_values.mean()
                    median_value = feature_values.median()
                    mean_median_gap = abs(mean_value - median_value)

                    st.metric("Range", f"{feature_values.max() - feature_values.min():.3f}")
                    st.metric("Standard deviation", f"{feature_values.std():.3f}")
                    st.metric("Mean–median difference", f"{mean_median_gap:.3f}")

                    if mean_value > median_value:
                        shape_note = "The mean is above the median. This can indicate a right-skewed distribution."
                    elif mean_value < median_value:
                        shape_note = "The mean is below the median. This can indicate a left-skewed distribution."
                    else:
                        shape_note = "The mean and median are equal, suggesting a balanced centre."
                    st.info(shape_note)

                with st.expander("View detailed statistics"):
                    stats = feature_values.describe().to_frame(
                        name="Value"
                    )
                    st.dataframe(stats, width="stretch")

            else:

                category_counts = (
                    feature_values.astype(str)
                    .value_counts()
                    .head(25)
                )

                fig_dist, ax_dist = plt.subplots(figsize=(10, 4.8))

                sns.barplot(
                    x=category_counts.index,
                    y=category_counts.values,
                    color="#087F8C",
                    ax=ax_dist,
                )

                ax_dist.set_title(
                    f"Top categories in {selected_feature}",
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                )
                ax_dist.set_xlabel(selected_feature, fontsize=10)
                ax_dist.set_ylabel("Number of records", fontsize=10)
                ax_dist.tick_params(axis="x", rotation=35, labelsize=9)
                ax_dist.tick_params(axis="y", labelsize=9)
                ax_dist.grid(axis="y", alpha=0.22)
                ax_dist.set_axisbelow(True)
                ax_dist.spines["top"].set_visible(False)
                ax_dist.spines["right"].set_visible(False)

                # Display the count above each category bar.
                for bar in ax_dist.patches:
                    height = bar.get_height()
                    ax_dist.annotate(
                        f"{int(height):,}",
                        (bar.get_x() + bar.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        xytext=(0, 3),
                        textcoords="offset points",
                        fontsize=8,
                    )

                fig_dist.tight_layout()
                st.pyplot(fig_dist, width="stretch")
                plt.close(fig_dist)

                st.dataframe(
                    category_counts.rename_axis("Category")
                    .reset_index(name="Records"),
                    width="stretch",
                    hide_index=True,
                )


    # ============================================================
    # TAB 4: CORRELATION
    # ============================================================
    with tab_correlation:
        st.subheader("Correlation heatmap")
        st.caption(
            "Correlation describes the strength and direction of a linear relationship. "
            "Red indicates positive correlation; blue indicates negative correlation. "
            "Correlation does not prove causation."
        )

        numeric_features = [
            column for column in feature_names
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        if len(numeric_features) > 10:
            selected_features = st.multiselect(
                "Choose 2–10 numerical features to compare",
                options=list(numeric_features),
                default=list(numeric_features[:8]),
                max_selections=10,
                key="correlation_features",
                help="For readability, the heatmap starts with eight features selected.",
            )

            show_all_features = st.checkbox(
                f"Show all {len(numeric_features)} numerical features (larger heatmap)",
                value=False,
                key="correlation_all_features",
            )

            if show_all_features:
                selected_features = list(numeric_features)
        else:
            selected_features = list(numeric_features)

        show_values = st.checkbox(
            "Show correlation values inside cells",
            value=False,
            key="correlation_values",
        )

        if len(selected_features) < 2:
            st.info("Select at least two features to display the heatmap.")
        else:
            correlation_matrix = df[selected_features].corr()
            n_features = len(selected_features)

            if n_features <= 5:
                fig_width = 5.5
                cell_label_size = 9
                tick_label_size = 9
            elif n_features <= 10:
                fig_width = 7.2
                cell_label_size = 7
                tick_label_size = 8
            else:
                fig_width = min(12, max(8, n_features * 0.34))
                cell_label_size = 5
                tick_label_size = 6

            fig_height = min(fig_width, max(4.5, n_features * 0.32))
            fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                correlation_matrix,
                annot=show_values,
                fmt=".2f",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.35,
                linecolor="white",
                cbar_kws={"shrink": 0.72, "label": "Correlation"},
                annot_kws={"size": cell_label_size},
                ax=ax_corr,
            )

            ax_corr.set_title(
                f"Feature correlations — {dataset_name}",
                fontsize=11,
                pad=12,
            )
            ax_corr.tick_params(
                axis="x",
                labelrotation=45,
                labelsize=tick_label_size,
            )
            ax_corr.tick_params(
                axis="y",
                labelrotation=0,
                labelsize=tick_label_size,
            )
            fig_corr.tight_layout()
            st.pyplot(fig_corr, width="content")
            plt.close(fig_corr)

            if n_features > 10:
                st.caption(
                    "Large heatmaps are harder to read. For a clearer comparison, "
                    "turn off 'Show all features' and select a smaller group."
                )

    # ============================================================
    # TAB 5: DATA QUALITY
    # ============================================================
    with tab_quality:
        st.subheader("Data quality")
        st.caption(
            "Check for missing values before using a dataset for analysis or model training."
        )

        missing_values = df.isna().sum()
        total_missing = int(missing_values.sum())
        columns_with_missing = int((missing_values > 0).sum())
        complete_records = int(df.notna().all(axis=1).sum())

        quality1, quality2, quality3 = st.columns(3)
        quality1.metric("Total missing values", f"{total_missing:,}")
        quality2.metric("Columns with missing values", columns_with_missing)
        quality3.metric("Complete records", f"{complete_records:,}")

        quality_summary = pd.DataFrame(
            {
                "Feature": df.columns,
                "Data type": df.dtypes.astype(str).values,
                "Missing values": missing_values.values,
                "Missing (%)": (missing_values / len(df) * 100).round(2).values,
            }
        )

        st.dataframe(
            quality_summary,
            width="stretch",
            hide_index=True,
            height=360,
        )

        if total_missing == 0:
            st.success("No missing values were found in this dataset.")
        else:
            st.warning(
                "Some values are missing. Review the table and decide how to handle "
                "them before training a model."
            )

        st.markdown("#### Data quality note")
        st.write(
            "A dataset with no missing values can still contain outliers, duplicate "
            "records, measurement errors, or features that are not useful for a model. "
            "Missing-value checks are one part of data quality assessment."
        )

    # ============================================================
    # 8. FOOTER
    # ============================================================
    st.divider()
    st.caption(
        "DataSphere • Dataset Explorer | Educational project using scikit-learn "
        "and UCI datasets. Dataset exploration supports learning and does not "
        "replace expert judgement."
    )


elif selected_page == "Machine Learning Lab":
    st.markdown(
        '<div class="ds-eyebrow">DATASPHERE • MODEL DEVELOPMENT</div>',
        unsafe_allow_html=True,
    )
    st.title("Machine Learning Lab")
    st.markdown(
        '<div class="ds-subtitle">Train, evaluate, and understand machine-learning models.</div>',
        unsafe_allow_html=True,
    )

    task_tab_reg, task_tab_class, task_tab_cluster = st.tabs(
        ["Regression", "Classification", "Clustering"]
    )

    with task_tab_reg:
        settings_col, evaluation_col = st.columns([0.9, 1.5], gap="large")
        with settings_col:
            st.markdown('<div class="ds-section-label">MODEL SETTINGS</div>', unsafe_allow_html=True)
            st.subheader("Regression")
            st.caption("Predict a numerical target using regression models.")

            reg_dataset = st.selectbox(
                "Dataset",
                ["Diabetes", "California Housing", "Wine Quality (UCI)"],
                key="reg_dataset",
            )
            reg_model_name = st.selectbox(
                "Model",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest"],
                key="reg_model",
            )
            test_size = st.slider(
                "Test Size",
                min_value=0.10,
                max_value=0.50,
                value=0.20,
                step=0.05,
                format="%.2f",
                key="reg_test_size",
            )
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="reg_random_state",
            )

        with evaluation_col:
            st.markdown('<div class="ds-section-label">MODEL EVALUATION</div>', unsafe_allow_html=True)
            if st.button("Train Model", type="primary", key="train_regression"):
                with st.spinner("Training regression model..."):
                    if reg_dataset == "Diabetes":
                        diabetes_data = load_diabetes()
                        X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
                        y = pd.Series(diabetes_data.target, name="target")
                    else:
                        reg_df, reg_features, _ = load_dataset(reg_dataset)
                        X = reg_df[reg_features].select_dtypes(include="number")
                        y = reg_df["target"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(test_size), random_state=int(random_state)
                    )
                    if reg_model_name == "Linear Regression":
                        model = LinearRegression()
                    elif reg_model_name == "Ridge Regression":
                        model = Ridge(alpha=1.0)
                    elif reg_model_name == "Lasso Regression":
                        model = Lasso(alpha=0.1, max_iter=10000)
                    elif reg_model_name == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=int(random_state))
                    else:
                        model = RandomForestRegressor(
                            n_estimators=100, random_state=int(random_state), n_jobs=-1
                        )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mse ** 0.5
                    r2 = r2_score(y_test, y_pred)

                st.success("Model trained successfully!")
                with st.container(key="regression_metrics"):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE", f"{mae:.2f}")
                    m2.metric("MSE", f"{mse:,.0f}")
                    m3.metric("RMSE", f"{rmse:.2f}")
                    m4.metric("R²", f"{r2:.3f}")

                # Actual vs Predicted visualization
                result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                fig, ax = plt.subplots(figsize=(5.2, 3.8))
                sns.scatterplot(data=result_df, x="Actual", y="Predicted", color="#087F8C", ax=ax)
                low = min(result_df["Actual"].min(), result_df["Predicted"].min())
                high = max(result_df["Actual"].max(), result_df["Predicted"].max())
                ax.plot([low, high], [low, high], linestyle="--", color="#C43D4B", label="Perfect prediction")
                ax.set_title("Actual vs Predicted")
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig, width="content")
                plt.close(fig)

                with st.expander("View prediction results"):
                    st.dataframe(result_df.reset_index(drop=True), width="stretch", hide_index=True)

    with task_tab_class:
        settings_col, evaluation_col = st.columns([0.9, 1.5], gap="large")
        with settings_col:
            st.markdown('<div class="ds-section-label">MODEL SETTINGS</div>', unsafe_allow_html=True)
            st.subheader("Classification")
            st.caption("Predict a category or class label from input features.")

            cls_dataset = st.selectbox(
                "Dataset",
                ["Iris", "Breast Cancer", "Wine (UCI)", "Heart Disease (UCI)", "Bank Marketing (UCI)"],
                key="cls_dataset",
            )
            cls_model_name = st.selectbox(
                "Model", ["Logistic Regression", "Decision Tree", "Random Forest"],
                key="cls_model",
            )
            cls_test_size = st.slider(
                "Test Size", 0.10, 0.50, 0.20, 0.05, key="cls_test_size"
            )
            cls_random_state = st.number_input(
                "Random State", min_value=0, max_value=999999, value=42, step=1,
                key="cls_random_state",
            )

        with evaluation_col:
            st.markdown('<div class="ds-section-label">MODEL EVALUATION</div>', unsafe_allow_html=True)
            if st.button("Train Model", type="primary", key="train_classification"):
                with st.spinner("Training classification model..."):
                    cls_df, cls_features, _ = load_dataset(cls_dataset)
                    X_raw = cls_df[cls_features].copy()
                    y = (cls_df["class_name"].astype(str) if "class_name" in cls_df.columns
                         else cls_df["target"].astype(str))

                    X_raw = X_raw.replace("?", pd.NA)
                    X = pd.get_dummies(X_raw, drop_first=False)
                    X = X.apply(pd.to_numeric, errors="coerce")
                    X = X.fillna(X.median()).fillna(0)

                    stratify_y = y if y.value_counts().min() >= 2 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(cls_test_size),
                        random_state=int(cls_random_state), stratify=stratify_y,
                    )

                    if cls_model_name == "Logistic Regression":
                        scaler = StandardScaler()
                        X_train_model = scaler.fit_transform(X_train)
                        X_test_model = scaler.transform(X_test)
                        model = LogisticRegression(max_iter=2000, random_state=int(cls_random_state))
                    elif cls_model_name == "Decision Tree":
                        X_train_model, X_test_model = X_train, X_test
                        model = DecisionTreeClassifier(random_state=int(cls_random_state))
                    else:
                        X_train_model, X_test_model = X_train, X_test
                        model = RandomForestClassifier(
                            n_estimators=100, random_state=int(cls_random_state), n_jobs=-1
                        )

                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_test_model)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.success("Classification model trained successfully!")
                # Classification metrics: compact custom cards in one row.
                # Prevents Streamlit's large st.metric font from truncating values.
                metric_values = [
                    ("Accuracy", accuracy),
                    ("Precision", precision),
                    ("Recall", recall),
                    ("F1-score", f1),
                ]
                metric_cols = st.columns(4, gap="small")
                for metric_col, (metric_label, metric_value) in zip(metric_cols, metric_values):
                    with metric_col:
                        st.markdown(
                            f"""<div style='background:#ffffff;border:1px solid #d9e2ec;border-radius:16px;padding:22px 16px;min-height:116px;'>
                            <div style='color:#718096;font-size:16px;margin-bottom:12px;'>{metric_label}</div>
                            <div style='color:#172b3a;font-size:30px;font-weight:400;line-height:1.2;white-space:nowrap;'>{metric_value:.3f}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                labels = sorted(y_test.unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 14}, linewidths=0.5,
                    linecolor="white", square=True, ax=ax_cm,
                    cbar=True,
                )
                ax_cm.set_xlabel("Predicted label", fontsize=12)
                ax_cm.set_ylabel("Actual label", fontsize=12)
                ax_cm.set_title("Confusion Matrix", fontsize=15, pad=12)
                ax_cm.tick_params(axis="both", labelsize=11)
                fig_cm.tight_layout()
                st.pyplot(fig_cm, width="stretch")
                plt.close(fig_cm)

                # Clean Classification Report table
                with st.expander("Classification Report", expanded=True):
                    report_dict = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                    report_rows = []
                    for class_label in sorted(y_test.unique()):
                        values = report_dict[str(class_label)]
                        report_rows.append({
                            "Class": str(class_label),
                            "Precision": values["precision"],
                            "Recall": values["recall"],
                            "F1-score": values["f1-score"],
                            "Support": int(values["support"]),
                        })

                    report_table = pd.DataFrame(report_rows)
                    st.dataframe(
                        report_table,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Precision": st.column_config.NumberColumn(format="%.2f"),
                            "Recall": st.column_config.NumberColumn(format="%.2f"),
                            "F1-score": st.column_config.NumberColumn(format="%.2f"),
                            "Support": st.column_config.NumberColumn(format="%d"),
                        },
                    )

                    report_accuracy = report_dict["accuracy"]
                    st.metric("Overall Accuracy", f"{report_accuracy:.2%}")
                    avg_rows = pd.DataFrame([
                        {
                            "Average": "Macro Average",
                            "Precision": report_dict["macro avg"]["precision"],
                            "Recall": report_dict["macro avg"]["recall"],
                            "F1-score": report_dict["macro avg"]["f1-score"],
                        },
                        {
                            "Average": "Weighted Average",
                            "Precision": report_dict["weighted avg"]["precision"],
                            "Recall": report_dict["weighted avg"]["recall"],
                            "F1-score": report_dict["weighted avg"]["f1-score"],
                        },
                    ])
                    st.dataframe(
                        avg_rows,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Precision": st.column_config.NumberColumn(format="%.2f"),
                            "Recall": st.column_config.NumberColumn(format="%.2f"),
                            "F1-score": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )
                with st.expander("View prediction results"):
                    st.dataframe(
                        pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True),
                        width="stretch", hide_index=True,
                    )

    with task_tab_cluster:
        settings_col, evaluation_col = st.columns([0.9, 1.5], gap="large")
        with settings_col:
            st.markdown('<div class="ds-section-label">MODEL SETTINGS</div>', unsafe_allow_html=True)
            st.subheader("Clustering")
            st.caption("Discover groups without using known class labels.")

            cluster_dataset = st.selectbox(
                "Dataset", ["Iris", "Wine (UCI)", "Breast Cancer"], key="cluster_dataset"
            )
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 3, key="cluster_k")
            cluster_random_state = st.number_input(
                "Random State", min_value=0, max_value=999999, value=42, step=1,
                key="cluster_random_state",
            )

        with evaluation_col:
            st.markdown('<div class="ds-section-label">MODEL EVALUATION</div>', unsafe_allow_html=True)
            if st.button("Run Clustering", type="primary", key="run_clustering"):
                with st.spinner("Running K-Means clustering..."):
                    cl_df, cl_features, _ = load_dataset(cluster_dataset)
                    X_raw = cl_df[cl_features].select_dtypes(include="number").copy()
                    X_raw = X_raw.fillna(X_raw.median()).fillna(0)
                    X_scaled = StandardScaler().fit_transform(X_raw)
                    kmeans = KMeans(n_clusters=int(n_clusters),
                                   random_state=int(cluster_random_state), n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    from sklearn.decomposition import PCA
                    coordinates = PCA(n_components=2, random_state=int(cluster_random_state)).fit_transform(X_scaled)
                    plot_df = pd.DataFrame({
                        "Component 1": coordinates[:, 0],
                        "Component 2": coordinates[:, 1],
                        "Cluster": cluster_labels.astype(str),
                    })

                st.success("Clustering completed!")
                # Larger K-Means visualization, consistent with Regression plots.
                fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5.5))
                sns.scatterplot(
                    data=plot_df,
                    x="Component 1",
                    y="Component 2",
                    hue="Cluster",
                    palette="tab10",
                    s=45,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.35,
                    ax=ax_cluster,
                )
                ax_cluster.set_title("K-Means Clusters (PCA)", fontsize=15, pad=12)
                ax_cluster.set_xlabel("Component 1", fontsize=12)
                ax_cluster.set_ylabel("Component 2", fontsize=12)
                ax_cluster.tick_params(axis="both", labelsize=10)
                ax_cluster.grid(alpha=0.18)
                ax_cluster.legend(
                    title="Cluster",
                    fontsize=10,
                    title_fontsize=10,
                    loc="best",
                    frameon=True,
                )
                fig_cluster.tight_layout()
                st.pyplot(fig_cluster, width="stretch")
                plt.close(fig_cluster)

                st.metric("Inertia", f"{kmeans.inertia_:.2f}")
                st.caption("Inertia is the sum of squared distances from samples to their assigned cluster centers.")
                with st.expander("View cluster assignments"):
                    assignment_df = cl_df.copy()
                    assignment_df["Cluster"] = cluster_labels
                    st.dataframe(assignment_df, width="stretch", hide_index=True)

