"""MongoDB Atlas connection, data overview, and model retraining utilities."""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_modules.custom_dataset_manager import get_custom_dataset_catalog
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DB_DEFAULT = "datasphere"
REQUIRED_COLLECTIONS = {
    "community_posts": "Community comments, replies, and reactions",
    "training_logs": "Retraining history and model evaluation",
    "dataset_uploads": "Uploaded dataset metadata",
    "custom_datasets": "User-uploaded CSV dataset catalog",
}


def _read_secret(name: str, default: str = "") -> str:
    """Read secrets on the server; never render values to the browser."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name.upper(), default) or "").strip()


@st.cache_resource(show_spinner=False, ttl=120)
def _mongo_client(uri: str):
    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    client.admin.command("ping")
    return client


def _prepare_training_data(
    load_dataset,
    dataset_name: str,
    include_categorical: bool = False,
    require_target: bool = True,
):
    frame, feature_names, _ = load_dataset(dataset_name)
    target = frame["target"].copy() if "target" in frame else None
    if require_target and target is None:
        raise ValueError("The selected dataset does not contain a target column.")

    raw_features = frame[feature_names].copy().replace("?", pd.NA)
    if include_categorical:
        # Match the preprocessing used by the existing Classification page.
        features = pd.get_dummies(raw_features, drop_first=False).apply(pd.to_numeric, errors="coerce")
    else:
        features = raw_features.select_dtypes(include=[np.number]).copy()
    if features.empty:
        raise ValueError("No numeric features are available for training.")

    if target is not None:
        usable = target.notna()
        features = features.loc[usable]
        target = target.loc[usable]

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    if len(features) < 10:
        raise ValueError("There are not enough valid rows for retraining.")
    return features, target


def _persist_model(db, model, metadata: dict[str, Any]) -> str:
    """Persist a fitted sklearn pipeline in GridFS and return its file id."""
    from gridfs import GridFS

    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    buffer.seek(0)
    fs = GridFS(db, collection="model_artifacts")
    file_id = fs.put(
        buffer.getvalue(),
        filename=f"{metadata['task']}_{metadata['dataset']}_{metadata['model_name']}.joblib",
        content_type="application/octet-stream",
        metadata={
            "task": metadata["task"],
            "dataset": metadata["dataset"],
            "model_name": metadata["model_name"],
            "created_at": metadata["created_at"],
        },
    )
    return str(file_id)


def _run_retraining(load_dataset, task: str, dataset_name: str, model_name: str,
                    test_size: float, random_state: int, n_clusters: int):
    X, y = _prepare_training_data(
        load_dataset,
        dataset_name,
        include_categorical=(task == "Classification"),
        require_target=(task != "Clustering"),
    )
    metrics: dict[str, float] = {}

    if task == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(X) < 10:
            raise ValueError("The regression target does not contain enough valid numeric values.")
        if model_name == "Linear Regression":
            estimator = LinearRegression()
        elif model_name == "Ridge Regression":
            estimator = Ridge(alpha=1.0)
        elif model_name == "Decision Tree":
            estimator = DecisionTreeRegressor(random_state=random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
        if model_name in {"Linear Regression", "Ridge Regression"}:
            model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        else:
            model = Pipeline([("model", estimator)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, prediction)),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_test, prediction)),
        }
        train_rows, test_rows = len(X_train), len(X_test)

    elif task == "Classification":
        y = y.astype(str)
        stratify = y if y.value_counts().min() >= 2 else None
        if model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=1500, class_weight="balanced")
            model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        elif model_name == "Decision Tree":
            model = Pipeline([("model", DecisionTreeClassifier(random_state=random_state, class_weight="balanced"))])
        else:
            model = Pipeline([("model", RandomForestClassifier(
                n_estimators=150, random_state=random_state, class_weight="balanced", n_jobs=-1
            ))])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = {
            "Accuracy": float(accuracy_score(y_test, prediction)),
            "Precision": float(precision_score(y_test, prediction, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, prediction, average="weighted", zero_division=0)),
            "F1-score": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
        }
        train_rows, test_rows = len(X_train), len(X_test)

    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ])
        labels = model.fit_predict(X)
        metrics = {"Inertia": float(model.named_steps["model"].inertia_)}
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            metrics["Silhouette"] = float(silhouette_score(model.named_steps["scaler"].transform(X), labels))
        train_rows, test_rows = len(X), 0

    return model, metrics, len(X), len(X.columns), train_rows, test_rows


def render_mongodb_retraining(load_dataset, dataset_tasks: dict[str, str] | None = None):
    """Render a retraining-only workspace; MongoDB remains the server-side store."""
    st.markdown('<div class="ds-eyebrow">NUSAINSIGHT • MODEL RETRAINING</div>', unsafe_allow_html=True)
    st.title("MongoDB & Retraining")
    st.markdown(
        '<div class="ds-subtitle">Retrain compatible models, compare evaluation results, and save model artifacts to MongoDB.</div>',
        unsafe_allow_html=True,
    )

    uri = _read_secret("mongo_uri") or _read_secret("MONGODB_URI")
    database_name = _read_secret("mongo_database") or os.getenv("MONGODB_DATABASE", DB_DEFAULT)
    client = None
    db = None
    if uri:
        try:
            client = _mongo_client(uri)
            db = client[database_name]
        except Exception:
            client = None
            db = None

    if db is None:
        st.warning(
            "MongoDB is not connected. Model training can be configured after the database connection is restored; "
            "training history and model artifacts require MongoDB."
        )
        st.caption("Check the server-side `mongo_uri` and `mongo_database` settings in `.streamlit/secrets.toml`.")
        return

    task = st.selectbox(
        "Learning task",
        ["Regression", "Classification", "Clustering"],
        key="mongo_retrain_task",
    )
    catalog = get_custom_dataset_catalog()
    tasks = dataset_tasks or {
        "Diabetes": "Regression",
        "California Housing": "Regression",
        "Wine Quality (UCI)": "Regression / Classification",
        "Breast Cancer": "Classification",
        "Iris": "Classification / Clustering",
        "Wine (UCI)": "Classification / Clustering",
        "Heart Disease (UCI)": "Classification",
        "Bank Marketing (UCI)": "Classification",
    }

    if task == "Clustering":
        datasets = [
            "Iris", "Wine (UCI)", "Breast Cancer", "Diabetes",
            "California Housing", "Wine Quality (UCI)",
            "Heart Disease (UCI)", "Bank Marketing (UCI)",
        ]
        datasets.extend(
            item["name"] for item in catalog
            if item.get("task") == "Clustering" and item.get("name") not in datasets
        )
        models = ["K-Means"]
    else:
        datasets = [
            name for name, task_label in tasks.items()
            if task.lower() in str(task_label).lower()
        ]
        datasets.extend(
            item["name"] for item in catalog
            if item.get("task") == task and item.get("name") not in datasets
        )
        if task == "Regression":
            models = ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"]
        else:
            models = ["Logistic Regression", "Decision Tree", "Random Forest"]

    if not datasets:
        st.info(f"No {task.lower()} datasets are available. Add a compatible dataset in Dataset Explorer first.")
        return

    settings_col, evaluation_col = st.columns([0.9, 1.4], gap="large")
    with settings_col:
        dataset_name = st.selectbox("Dataset", datasets, key=f"mongo_retrain_dataset_{task}")
        model_name = st.selectbox("Model", models, key=f"mongo_retrain_model_{task}")
        random_state = int(st.number_input("Random seed", 0, 999999, 42, key="mongo_retrain_seed"))
        if task == "Clustering":
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 3, key="mongo_retrain_k")
            test_size = 0.0
        else:
            n_clusters = 3
            test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05, key="mongo_retrain_test_size")
        run_training = st.button(
            "Run retraining and save model",
            type="primary",
            use_container_width=True,
            key="mongo_run_retraining",
        )

    with evaluation_col:
        st.subheader("Evaluation")
        previous = db["training_logs"].find_one(
            {"task": task, "dataset": dataset_name, "status": "completed"},
            sort=[("created_at", -1)],
        )
        if previous and previous.get("metrics"):
            st.caption("Previous saved result for this dataset and task")
            before_metrics = previous.get("metrics", {})
            before_rows = [
                {"Stage": "Before retraining", "Metric": key, "Value": value}
                for key, value in before_metrics.items()
                if isinstance(value, (int, float))
            ]
            if before_rows:
                st.dataframe(pd.DataFrame(before_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No previous training result is available for this dataset. The next run will become the baseline.")

    if run_training:
        try:
            with st.spinner("Preparing data, training the model, and saving the artifact..."):
                model, metrics, sample_count, feature_count, train_rows, test_rows = _run_retraining(
                    load_dataset, task, dataset_name, model_name,
                    float(test_size), random_state, int(n_clusters),
                )
                now = datetime.now(timezone.utc)
                metadata = {
                    "task": task.lower().replace(" ", "_"),
                    "dataset": dataset_name.replace(" ", "_"),
                    "model_name": model_name.replace(" ", "_"),
                    "created_at": now,
                }
                artifact_id = _persist_model(db, model, metadata)
                log = {
                    "created_at": now,
                    "task": task,
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "sample_count": int(sample_count),
                    "feature_count": int(feature_count),
                    "train_rows": int(train_rows),
                    "test_rows": int(test_rows),
                    "test_size": float(test_size),
                    "random_state": int(random_state),
                    "metrics": metrics,
                    "previous_metrics": (previous or {}).get("metrics", {}),
                    "artifact_file_id": artifact_id,
                    "status": "completed",
                }
                db["training_logs"].insert_one(log)
            st.success("Retraining completed. The model artifact and evaluation log were saved to MongoDB.")
            st.rerun()
        except Exception as exc:
            st.error(f"Retraining failed: {type(exc).__name__}: {exc}")

    # Show the latest result after reruns, including a true before/after comparison.
    latest = db["training_logs"].find_one(
        {"task": task, "dataset": dataset_name, "status": "completed"},
        sort=[("created_at", -1)],
    )
    if latest:
        st.markdown("### Latest retraining result")
        current_metrics = latest.get("metrics", {}) or {}
        metric_cols = st.columns(max(1, len(current_metrics)))
        for col, (metric_name, metric_value) in zip(metric_cols, current_metrics.items()):
            with col:
                value = f"{metric_value:.4f}" if isinstance(metric_value, (float, int)) else str(metric_value)
                st.metric(metric_name, value)

        previous_metrics = latest.get("previous_metrics", {}) or {}
        comparable = [key for key in current_metrics if key in previous_metrics]
        if comparable:
            comparison_rows = []
            for key in comparable:
                before_value = previous_metrics[key]
                after_value = current_metrics[key]
                delta = after_value - before_value
                comparison_rows.append({
                    "Metric": key,
                    "Before": before_value,
                    "After": after_value,
                    "Change": delta,
                })
            st.markdown("**Before vs. after retraining**")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        elif not previous_metrics:
            st.caption("This is the first saved run for this dataset and task; no earlier metrics are available for comparison.")

        st.caption(
            f"Model: {latest.get('model_name', '—')} · Samples: {latest.get('sample_count', 0):,} · "
            f"Features: {latest.get('feature_count', 0):,} · Artifact saved in MongoDB GridFS."
        )

    st.divider()
    st.subheader("Retraining history")
    try:
        logs = list(
            db["training_logs"].find(
                {},
                {"_id": 0, "created_at": 1, "dataset": 1, "task": 1,
                 "model_name": 1, "metrics": 1, "status": 1},
            ).sort("created_at", -1)
        )
        if logs:
            rows = []
            for item in logs:
                metric_values = item.get("metrics", {}) or {}
                rows.append({
                    "Created at": item.get("created_at"),
                    "Dataset": item.get("dataset"),
                    "Task": item.get("task"),
                    "Model": item.get("model_name"),
                    "Metrics": ", ".join(
                        f"{key}: {value:.4g}" if isinstance(value, (int, float)) else f"{key}: {value}"
                        for key, value in metric_values.items()
                    ),
                    "Status": item.get("status", "—"),
                })
            # The MongoDB cursor is intentionally unbounded; the fixed-height table
            # presents the newest five rows initially and scrolls to older records.
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
                height=230,
            )
        else:
            st.info("No retraining history yet. Run the first model to create a record.")
    except Exception as exc:
        st.warning(f"Training history is temporarily unavailable ({type(exc).__name__}).")

    st.caption("Model artifacts and evaluation logs are stored server-side in MongoDB. Credentials are never displayed.")
