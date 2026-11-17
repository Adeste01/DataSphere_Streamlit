import streamlit as st
import base64
from streamlit_cookies_controller import CookieController
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

# COMMUNITY SENTIMENT (transparent keyword baseline)
def analyze_community_sentiment(text):
    """Bilingual English/Indonesian keyword sentiment baseline with simple negation handling."""
    import re

    positive_words = {
        "good", "great", "excellent", "amazing", "awesome", "love", "like",
        "helpful", "useful", "wonderful", "fantastic", "easy", "enjoy",
        "thank", "thanks", "appreciate", "happy", "clear", "interesting",
        "best", "perfect", "nice", "support", "cool", "wow", "extraordinary",
        "brilliant", "outstanding", "impressive", "fantastic", "top", "ok", "okay",
        "oke", "cukup", "recommended", "recommend", "valuable", "friendly",
        "bagus", "baik", "hebat", "mantap", "keren", "membantu",
        "bermanfaat", "menyenangkan", "puas", "terima kasih", "mudah",
        "luar biasa", "suka", "senang", "jelas"
    }
    negative_words = {
        "bad", "poor", "terrible", "awful", "hate", "dislike", "difficult",
        "confusing", "hard", "wrong", "problem", "issue", "broken", "slow",
        "disappointing", "unhelpful", "worse", "worst", "annoying", "fail",
        "buruk", "jelek", "parah", "kecewa", "sulit", "bingung",
        "mengecewakan", "tidak membantu", "tidak bagus", "kurang",
        "gagal", "lambat", "rusak", "payah", "benci"
    }

    normalized = (text or "").lower()
    # Count multi-word phrases first, then remove them to avoid double counting.
    positive_phrases = ["terima kasih", "luar biasa"]
    negative_phrases = ["tidak membantu", "tidak bagus"]
    pos = sum(normalized.count(phrase) for phrase in positive_phrases)
    neg = sum(normalized.count(phrase) for phrase in negative_phrases)
    for phrase in positive_phrases + negative_phrases:
        normalized = normalized.replace(phrase, " ")

    tokens = re.findall(r"\b[a-zA-ZÀ-ÿ']+\b", normalized)
    pos += sum(token in positive_words for token in tokens)
    neg += sum(token in negative_words for token in tokens)

    # Simple negation handling: "not good"/"tidak bagus" should reverse polarity.
    negators = {"not", "no", "never", "tidak", "tak", "kurang"}
    for i, token in enumerate(tokens[:-1]):
        if token in negators:
            next_token = tokens[i + 1]
            if next_token in positive_words:
                pos -= 1
                neg += 1
            elif next_token in negative_words:
                neg -= 1
                pos += 1

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"

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
        ["Home", "Dataset Explorer", "Machine Learning Lab", "Community", "About & Methodology"],
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



# ============================================================
# 8. COMMUNITY
# ============================================================
elif selected_page == "Community":
    import datetime
    import uuid
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    st.markdown('<div class="ds-eyebrow">DATASPHERE • COMMUNITY</div>', unsafe_allow_html=True)
    st.title("💬 Community Hub")
    st.markdown(
        '<div class="ds-subtitle">Share your thoughts, ask questions, and react to comments from other visitors.</div>',
        unsafe_allow_html=True,
    )

    # MongoDB connection: credentials are kept in .streamlit/secrets.toml.
    if "community_user_reactions" not in st.session_state:
        st.session_state.community_user_reactions = {}
    # Persistent anonymous visitor identity (no login required).
    cookie_controller = CookieController()
    # Read the cookie without calling a nonexistent ready() method.
    # CookieController exposes get/set; use a stable session fallback while
    # the browser component initializes, then retain the cookie when available.
    visitor_id = cookie_controller.get("datasphere_visitor_id")
    if not visitor_id:
        visitor_id = st.session_state.get("community_visitor_id")
    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        st.session_state["community_visitor_id"] = visitor_id
        cookie_controller.set(
            "datasphere_visitor_id",
            visitor_id,
            max_age=60 * 60 * 24 * 365,
        )
    else:
        st.session_state["community_visitor_id"] = visitor_id
    try:
        mongo_client = MongoClient(
            st.secrets["mongo_uri"],
            serverSelectionTimeoutMS=5000,
        )
        mongo_client.admin.command("ping")
        community_collection = mongo_client["datasphere"]["community_posts"]
        posts = list(community_collection.find().sort("created_at", -1))
        for item in posts:
            item["id"] = str(item["_id"])
            item.setdefault("replies", [])
            item.setdefault("likes", 0)
            item.setdefault("dislikes", 0)
    except Exception as mongo_error:
        st.error("Unable to connect to MongoDB. Please check .streamlit/secrets.toml and your Atlas Network Access settings.")
        st.caption(f"Connection detail: {mongo_error}")
        st.stop()

    # Sentiment overview
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for item in posts:
        label = analyze_community_sentiment(item.get("text", ""))
        item["sentiment"] = label
        sentiment_counts[label] += 1
    total_comments = len(posts)
    total_likes = sum(p.get("likes", 0) for p in posts)
    total_dislikes = sum(p.get("dislikes", 0) for p in posts)
    positive_pct = (sentiment_counts["Positive"] / total_comments * 100) if total_comments else 0

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Comments", total_comments)
    metric2.metric("Total Likes", total_likes)
    metric3.metric("Total Dislikes", total_dislikes)
    metric4.metric("Positive Sentiment", f"{positive_pct:.0f}%")

    st.divider()
    form_col, sentiment_col = st.columns([1.35, 1], gap="large")
    with form_col:
        st.subheader("Share a Comment")
        with st.form("community_post_form", clear_on_submit=True):
            author = st.text_input("Your Name", placeholder="Anonymous Visitor", max_chars=40)
            post_text = st.text_area(
                "Your Comment",
                placeholder="Share your thoughts, suggestions, or feedback...",
                max_chars=300,
                height=95,
            )
            st.caption(f"{len(post_text)}/300")
            submitted = st.form_submit_button("Post Comment", type="primary", use_container_width=True)
            if submitted:
                if not post_text.strip():
                    st.warning("Please write a comment before posting.")
                else:
                    new_post = {
                        "owner_id": visitor_id,
                        "author": author.strip() or "Anonymous Visitor",
                        "text": post_text.strip(),
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "likes": 0, "dislikes": 0, "replies": [],
                        "sentiment": analyze_community_sentiment(post_text.strip()),
                    }
                    community_collection.insert_one(new_post)
                    st.success("Comment posted!")
                    st.rerun()

    with sentiment_col:
        st.subheader("Community Sentiment")
        import matplotlib.pyplot as plt
        vals = [sentiment_counts["Positive"], sentiment_counts["Neutral"], sentiment_counts["Negative"]]
        labels = ["Positive", "Neutral", "Negative"]
        colors = ["#0F9D8A", "#9CA3AF", "#D66AC3"]

        if sum(vals) > 0:
            fig, ax = plt.subplots(figsize=(4.2, 2.8))
            ax.pie(vals, labels=None, colors=colors, startangle=90,
                   wedgeprops={"width": 0.42, "edgecolor": "white"})
            ax.legend(labels=[f"{n}: {v} ({v/sum(vals)*100:.0f}%)" for n, v in zip(labels, vals)],
                      loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            # Clean empty state: no Matplotlib axes or placeholder chart.
            st.markdown(
                """
                <div style="background:#ffffff; border:1px solid #e2e8f0;
                            border-radius:14px; padding:34px 18px; text-align:center;
                            min-height:190px; display:flex; flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:34px; margin-bottom:8px;">📊</div>
                    <div style="font-size:17px; font-weight:600; color:#243b4a;">
                        Sentiment insights will appear here
                    </div>
                    <div style="font-size:13px; color:#718096; margin-top:6px;">
                        Post a comment to see the community sentiment breakdown.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        sort_option = st.selectbox("Sort by", ["Newest First", "Oldest First", "Most Likes"], key="community_sort")

    st.divider()
    heading_col, search_col = st.columns([1.15, 1.0], vertical_alignment="center")
    with heading_col:
        st.subheader("Latest Discussions")
    with search_col:
        search_query = st.text_input(
            "Search comments",
            placeholder="🔎  Search comments...",
            label_visibility="collapsed",
            key="community_search",
        )

    if not posts:
        st.info("No comments yet. Be the first to start a discussion!")
    else:
        display_posts = list(posts)

        # Case-insensitive keyword search in the main comment text.
        if search_query.strip():
            query = search_query.strip().lower()
            display_posts = [
                post for post in display_posts
                if query in str(post.get("text", "")).lower()
            ]

        if sort_option == "Oldest First":
            display_posts.reverse()
        elif sort_option == "Most Likes":
            display_posts.sort(key=lambda p: p.get("likes", 0), reverse=True)

        if search_query.strip() and not display_posts:
            st.warning(f"No comments found for '{search_query.strip()}'.")
        sentiment_badges = {"Positive": "🟢 Positive", "Neutral": "⚪ Neutral", "Negative": "🔴 Negative"}
        for post in display_posts:
            post_id = post.setdefault("id", str(uuid.uuid4()))
            with st.container(border=True):
                # Compact header: author/time on the left, owner controls side-by-side on the right.
                st.markdown(
                    """<style>
                    div[data-testid="stButton"] button[kind="secondary"] {
                        min-width: 2.8rem;
                        padding: 0.35rem 0.55rem;
                        line-height: 1.25;
                    }
                    </style>""",
                    unsafe_allow_html=True,
                )
                head_col, edit_col, delete_col = st.columns([10, 1.15, 1.15], vertical_alignment="center")
                with head_col:
                    avatar = (post.get("author") or "A")[0].upper()
                    st.markdown(
                        f"<span style='display:inline-flex;background:#087F8C;color:white;"
                        f"border-radius:50%;width:34px;height:34px;align-items:center;"
                        f"justify-content:center;font-weight:700;margin-right:8px;'>{avatar}</span>"
                        f"<strong>{post.get('author','Anonymous Visitor')}</strong>"
                        f"<span style='color:#81919A;font-size:.78rem;margin-left:8px;'>"
                        f"{post.get('created_at','')}</span>",
                        unsafe_allow_html=True,
                    )
                if post.get("owner_id") == visitor_id:
                    with edit_col:
                        if st.button("✏️", key=f"edit_open_{post_id}", help="Edit your comment"):
                            st.session_state[f"edit_mode_{post_id}"] = True
                            st.rerun()
                    with delete_col:
                        if st.button("🗑️", key=f"delete_{post_id}", help="Delete your comment"):
                            st.session_state[f"confirm_delete_{post_id}"] = True
                            st.rerun()

                # Ask for confirmation before permanently removing a comment.
                if st.session_state.get(f"confirm_delete_{post_id}", False):
                    st.warning("Are you sure you want to delete this comment? This action cannot be undone.")
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("Yes, delete", key=f"confirm_yes_{post_id}", type="primary"):
                            community_collection.delete_one({"_id": __import__("bson").ObjectId(post_id), "owner_id": visitor_id})
                            st.session_state.community_user_reactions.pop(post_id, None)
                            st.session_state.pop(f"edit_mode_{post_id}", None)
                            st.session_state.pop(f"confirm_delete_{post_id}", None)
                            st.rerun()
                    with cancel_col:
                        if st.button("Cancel", key=f"confirm_no_{post_id}"):
                            st.session_state.pop(f"confirm_delete_{post_id}", None)
                            st.rerun()

                # Edit form appears only for the owner in this browser session.
                if st.session_state.get(f"edit_mode_{post_id}", False):
                    with st.form(f"edit_form_{post_id}"):
                        edited_text = st.text_area("Edit comment", value=post.get("text", ""), max_chars=300, key=f"edit_text_{post_id}")
                        save_edit = st.form_submit_button("Save changes")
                        cancel_edit = st.form_submit_button("Cancel")
                        if save_edit:
                            if edited_text.strip():
                                updated_text = edited_text.strip()
                                updated_sentiment = analyze_community_sentiment(updated_text)
                                community_collection.update_one(
                                    {"_id": __import__("bson").ObjectId(post_id), "owner_id": visitor_id},
                                    {"$set": {"text": updated_text, "sentiment": updated_sentiment,
                                              "edited_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}}
                                )
                                st.session_state[f"edit_mode_{post_id}"] = False
                                st.success("Comment updated.")
                                st.rerun()
                            else:
                                st.warning("Comment cannot be empty.")
                        if cancel_edit:
                            st.session_state[f"edit_mode_{post_id}"] = False
                            st.rerun()

                st.markdown(f"<div style='margin:5px 0 3px 0;font-size:1rem;'>{post.get('text','')}</div>", unsafe_allow_html=True)
                post_sentiment = analyze_community_sentiment(post.get("text", ""))
                post["sentiment"] = post_sentiment
                st.caption(f"Sentiment: {sentiment_badges[post_sentiment]}")

                # Compact reaction/reply row, with one reaction per visitor.
                reaction_col1, reaction_col2, reply_col, spacer_col = st.columns([1, 1, 1.25, 5])
                reactions = st.session_state.community_user_reactions
                current_reaction = reactions.get(post_id)
                with reaction_col1:
                    like_label = f"👍 {post.get('likes', 0)}" + (" ✓" if current_reaction == "like" else "")
                    if st.button(like_label, key=f"like_{post_id}"):
                        from bson import ObjectId
                        if current_reaction == "like":
                            community_collection.update_one({"_id": ObjectId(post_id)}, {"$inc": {"likes": -1}})
                            reactions.pop(post_id, None)
                        else:
                            increments = {"likes": 1}
                            if current_reaction == "dislike":
                                increments["dislikes"] = -1
                            community_collection.update_one({"_id": ObjectId(post_id)}, {"$inc": increments})
                            reactions[post_id] = "like"
                        st.rerun()
                with reaction_col2:
                    dislike_label = f"👎 {post.get('dislikes', 0)}" + (" ✓" if current_reaction == "dislike" else "")
                    if st.button(dislike_label, key=f"dislike_{post_id}"):
                        from bson import ObjectId
                        if current_reaction == "dislike":
                            community_collection.update_one({"_id": ObjectId(post_id)}, {"$inc": {"dislikes": -1}})
                            reactions.pop(post_id, None)
                        else:
                            increments = {"dislikes": 1}
                            if current_reaction == "like":
                                increments["likes"] = -1
                            community_collection.update_one({"_id": ObjectId(post_id)}, {"$inc": increments})
                            reactions[post_id] = "dislike"
                        st.rerun()
                reply_key = f"reply_open_{post_id}"
                if reply_key not in st.session_state:
                    st.session_state[reply_key] = False
                with reply_col:
                    if st.button(f"↩ Reply ({len(post.get('replies', []))})", key=f"reply_toggle_{post_id}"):
                        st.session_state[reply_key] = not st.session_state[reply_key]
                        st.rerun()

                if post.get("replies"):
                    with st.container():
                        toggle_label = ("▾ Hide Replies" if st.session_state.get(f"replies_view_{post_id}", False)
                                        else f"▸ View Replies ({len(post.get('replies', []))})")
                        if st.button(toggle_label, key=f"replies_toggle_{post_id}"):
                            st.session_state[f"replies_view_{post_id}"] = not st.session_state.get(f"replies_view_{post_id}", False)
                            st.rerun()

                # Replies are hidden by default and expanded only on visitor request.
                replies = post.setdefault("replies", [])
                replies_view_key = f"replies_view_{post_id}"
                if replies_view_key not in st.session_state:
                    st.session_state[replies_view_key] = False
                if replies and st.session_state[replies_view_key]:
                    st.markdown("**Replies**")
                    for reply in replies:
                        st.markdown(
                            f"<div style='margin-left:18px;padding:5px 10px;"
                            f"border-left:2px solid #087F8C;font-size:.9rem;'>"
                            f"<strong>{reply.get('author','Anonymous Visitor')}</strong> "
                            f"<span style='color:#81919A;font-size:.75rem;'>{reply.get('created_at','')}</span><br>"
                            f"{reply.get('text','')}</div>",
                            unsafe_allow_html=True,
                        )
                if st.session_state[reply_key]:
                    # A reply form without its own border avoids nested card borders
                    # colliding with the outer comment container.
                    st.markdown(
                        "<div style='margin:12px 0 4px 18px;padding:12px 14px;"
                        "border-left:3px solid #087F8C;background:rgba(8,127,140,.045);"
                        "border-radius:0 10px 10px 0;'>"
                        "<span style='font-weight:600;color:#087F8C;'>Write a reply</span></div>",
                        unsafe_allow_html=True,
                    )
                    with st.form(f"reply_form_{post_id}", clear_on_submit=True, border=False):
                        reply_author = st.text_input("Your Name", placeholder="Anonymous Visitor", max_chars=40, key=f"reply_author_{post_id}")
                        reply_text = st.text_area("Your Reply", placeholder="Write a reply...", max_chars=300, height=65, key=f"reply_text_{post_id}")
                        publish_col, cancel_col, _ = st.columns([1.2, 1, 5])
                        with publish_col:
                            publish_reply = st.form_submit_button("Publish Reply", type="primary")
                        with cancel_col:
                            cancel_reply = st.form_submit_button("Cancel")
                        if cancel_reply:
                            st.session_state[reply_key] = False
                            st.rerun()
                        if publish_reply:
                            if not reply_text.strip():
                                st.warning("Please write a reply.")
                            else:
                                new_reply = {
                                    "author": reply_author.strip() or "Anonymous Visitor",
                                    "text": reply_text.strip(),
                                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                }
                                from bson import ObjectId
                                community_collection.update_one(
                                    {"_id": ObjectId(post_id)},
                                    {"$push": {"replies": new_reply}}
                                )
                                st.session_state[reply_key] = False
                                st.success("Reply published!")
                                st.rerun()

    st.caption("Community comments, replies, and reactions are stored in MongoDB Atlas.")


elif selected_page == "About & Methodology":
    st.title("📖 About & Methodology")
    st.caption("Learn more about this project, our approach, and the technologies used.")

    about_tab, methodology_tab, architecture_tab, group_tab = st.tabs([
        "About the Project",
        "Methodology",
        "System Architecture",
        "Group & Acknowledgement",
    ])

    with about_tab:
        st.header("About DataSphere")
        st.write(
            "DataSphere is an interactive Data Science platform designed to help users "
            "explore datasets, experiment with Machine Learning algorithms, evaluate "
            "model performance, and share insights with a community of learners. "
            "Our goal is to make data-driven exploration accessible and engaging."
        )
        feature_cols = st.columns(4)
        feature_items = [
            ("⚙️", "Data Exploration", "Understand datasets through clear visualizations."),
            ("🧠", "Machine Learning", "Experiment with different algorithms."),
            ("💡", "Interactive Learning", "Explore how data and models work together."),
            ("👥", "Community", "Share feedback and learn from others."),
        ]
        for col, (emoji, title, desc) in zip(feature_cols, feature_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {emoji}")
                    st.markdown(f"**{title}**")
                    st.caption(desc)

        st.subheader("Why We Built This")
        left, right = st.columns([1.2, 1])
        with left:
            st.write(
                "We believe data can help us understand the world better. "
                "DataSphere combines interactive tools, visualizations, and model "
                "evaluation to support learning and exploration across different backgrounds."
            )
        with right:
            st.markdown("#### Project Information")
            st.table({
                "Item": ["Course", "Study Program", "University", "Academic Year"],
                "Information": [
                    "Artificial Intelligence",
                    "Master of Data Science",
                    "Universitas Kristen Satya Wacana",
                    "2026/2027",
                ],
            })

    with methodology_tab:
        st.header("Data Science Workflow")
        st.write(
            "DataSphere follows an interactive Data Science workflow that allows users "
            "to explore datasets, build Machine Learning models, evaluate results, "
            "and communicate insights."
        )

        workflow = [
            ("01", "Dataset Selection", "Choose a dataset available in DataSphere, such as Iris, Breast Cancer, Diabetes, or California Housing."),
            ("02", "Data Exploration", "Inspect dataset dimensions, features, sample records, distributions, correlations, and data quality."),
            ("03", "Model Development", "Select a suitable task—Regression, Classification, or Clustering—and configure the available model options"),
            ("04", "Model Evaluation", "Review appropriate evaluation metrics and visualizations to understand model performance"),
            ("05", "Interpretation & Sharing", "Interpret the results and share observations or feedback through the Community Hub"),
        ]

        for number, title, description in workflow:
            with st.container(border=True):
                left, right = st.columns([0.12, 0.88])
                with left:
                    st.markdown(f"### :blue[{number}]")
                with right:
                    st.markdown(f"**{title}**")
                    st.write(description)

        st.caption(
            "Note: The workflow and available metrics depend on the selected dataset "
            "and Machine Learning task."
        )

    with architecture_tab:
        st.header("DataSphere System Architecture")
        st.write(
            "DataSphere is built as a Streamlit application. Its interface connects "
            "the user to interactive modules, Python-based data and Machine Learning "
            "processing, visual outputs, and MongoDB Atlas for Community data."
        )

        def architecture_box(title, detail, color):
            st.markdown(
                f"""
                <div style="background:{color}; padding:16px 18px; border-radius:12px;
                            border:1px solid rgba(100,116,139,.25); text-align:center; margin:4px 0;">
                    <div style="font-weight:700; font-size:1.05rem;">{title}</div>
                    <div style="font-size:.9rem; margin-top:5px;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        architecture_box("User Interface", "Streamlit Web Application", "#E7F1FF")
        st.markdown("<div style='text-align:center;font-size:24px;'>↓</div>", unsafe_allow_html=True)
        architecture_box(
            "Application Modules",
            "Home  •  Dataset Explorer  •  Machine Learning Lab  •  Community  •  About & Methodology",
            "#E8F5E9",
        )
        st.markdown("<div style='text-align:center;font-size:24px;'>↓</div>", unsafe_allow_html=True)

        processing_col, visualization_col = st.columns(2)
        with processing_col:
            architecture_box("Data & ML Processing", "Python • Pandas • NumPy • Scikit-learn", "#F3E8FF")
        with visualization_col:
            architecture_box("Visualization", "Streamlit • Matplotlib • Seaborn", "#FFF3E0")

        st.markdown("<div style='text-align:center;font-size:24px;'>↓</div>", unsafe_allow_html=True)
        architecture_box(
            "Database Layer",
            "MongoDB Atlas — Community comments, replies, and reactions",
            "#E0F2F1",
        )

        st.subheader("Architecture Components")
        component_col1, component_col2 = st.columns(2)
        with component_col1:
            st.markdown("**Application & Processing**")
            st.write("- Streamlit provides the interactive web interface.")
            st.write("- Python handles application logic and data processing.")
            st.write("- Scikit-learn provides Machine Learning algorithms and evaluation.")
            st.write("- Pandas and NumPy support data manipulation.")
        with component_col2:
            st.markdown("**Visualization & Storage**")
            st.write("- Matplotlib and Seaborn support data visualizations.")
            st.write("- Streamlit displays charts and model results.")
            st.write("- MongoDB Atlas stores Community posts, replies, and reactions.")

    with group_tab:
        st.header("Our Team")
        st.write(
            "DataSphere was developed collaboratively by a team of three students "
            "with different academic and technical backgrounds."
        )

        # Team profile cards: equal height, matching layout, and circular portraits
        adeste_photo = "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"
        octaviana_photo = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAJkAZgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDmqKKK/tU/icKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiikZlUZZgAO5NGwJN6IWisbUfGHhvSwf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"
        team_cards = [
            ("ADESTE CHARISMA LUMENVITHA ADHI", "Project Development",
             "Contributes to the design, development, integration, and deployment of the DataSphere application.",
             adeste_photo),
            ("OCTAVIANA AYU HARINI", "Data Science & Application Development",
             "Contributes to data processing, modeling, feature development, and the overall content of the DataSphere application.",
             octaviana_photo),
        ]
        card_cols = st.columns(2, gap="large")
        for col, (member, role, contribution, photo) in zip(card_cols, team_cards):
            with col:
                st.markdown(
                    f"""
                    <div style="min-height:365px; height:365px; box-sizing:border-box;
                        border:1px solid #d5e0e7; border-radius:16px; padding:28px 24px;
                        background:linear-gradient(180deg,#ffffff 0%,#f7fbfd 100%);
                        display:flex; flex-direction:column; align-items:center; text-align:center;">
                        <img src="data:image/jpeg;base64,{photo}" style="width:92px;height:92px;
                            border-radius:50%;object-fit:cover;border:3px solid #d9eef2;margin-bottom:18px;">
                        <div style="font-size:20px;font-weight:700;color:#243f50;min-height:56px;
                            display:flex;align-items:center;justify-content:center;">{member}</div>
                        <div style="display:inline-block;background:#e5f5f7;color:#176477;
                            border-radius:20px;padding:7px 15px;font-size:14px;font-weight:600;
                            margin:12px 0 18px;">{role}</div>
                        <div style="border-top:1px solid #dce7ec;width:100%;padding-top:18px;
                            color:#526574;font-size:15px;line-height:1.6;">{contribution}</div>
                    </div>
                    """.replace("{photo}", photo).replace("{member}", member).replace("{role}", role).replace("{contribution}", contribution),
                    unsafe_allow_html=True
                )

        st.divider()
        st.header("Acknowledgement")
        st.write(
            "We would like to express our sincere gratitude to our lecturers for their "
            "guidance, feedback, and valuable knowledge, and to our team for its "
            "collaboration and commitment in developing DataSphere."
        )
        st.markdown("**Lecturers & Acknowledgements**")
        lecturers = [
            "Prof. Hanna Arini Parhusip, M.Sc.nat",
            "Dr. Suryasatriya Trihandaru, M.Sc.nat",
            "Dr. Bambang Susanto",
            "Prof. Dr. Eko Soediyono, M.Kom",
            "Denny Indrajaya, S.Si., M.Si.D",
        ]
        for lecturer in lecturers:
            st.markdown(f"- {lecturer}")
