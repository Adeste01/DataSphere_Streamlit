"""Custom CSV dataset management for NusaInsight.

Uploads are validated in Streamlit, persisted in MongoDB, and exposed through
load_custom_dataset() so Dataset Explorer, Machine Learning Lab, Home, and
MongoDB retraining can reuse the same dataset catalog.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st


DB_DEFAULT = "datasphere"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 100


def _read_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name.upper(), default)
    return str(value or os.getenv(name.upper(), default) or "").strip()


@st.cache_resource(show_spinner=False, ttl=120)
def _mongo_client(uri: str):
    from pymongo import MongoClient

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        retryWrites=True,
    )
    client.admin.command("ping")
    return client


def get_database():
    """Return the configured MongoDB database, or None when unavailable."""
    uri = _read_secret("mongo_uri") or _read_secret("MONGODB_URI")
    database_name = (
        _read_secret("mongo_database")
        or _read_secret("MONGODB_DATABASE")
        or DB_DEFAULT
    )
    if not uri:
        return None
    try:
        return _mongo_client(uri)[database_name]
    except Exception:
        return None


def get_custom_dataset_catalog() -> list[dict[str, Any]]:
    """Read custom dataset metadata; safe to call when MongoDB is unavailable."""
    db = get_database()
    if db is None:
        return []
    try:
        return list(
            db["custom_datasets"].find(
                {},
                {
                    "_id": 0,
                    "name": 1,
                    "name_key": 1,
                    "task": 1,
                    "target_column": 1,
                    "feature_columns": 1,
                    "row_count": 1,
                    "column_count": 1,
                    "numeric_feature_count": 1,
                    "filename": 1,
                    "collection_name": 1,
                    "created_at": 1,
                },
            ).sort("created_at", -1)
        )
    except Exception:
        return []


def load_custom_dataset(dataset_name: str):
    """Load one saved custom dataset in the same format as the app's loader.

    Returns (DataFrame, feature_names, target_names). Classification adds
    target and class_name; regression adds numeric target; clustering is
    unsupervised and therefore has no target column.
    """
    db = get_database()
    if db is None:
        raise ConnectionError(
            "MongoDB is not connected. Connect MongoDB first."
        )

    metadata = db["custom_datasets"].find_one(
        {"name": dataset_name},
        {
            "_id": 0,
            "collection_name": 1,
            "task": 1,
            "target_column": 1,
            "feature_columns": 1,
        },
    )
    if not metadata:
        raise ValueError(f"Custom dataset '{dataset_name}' was not found.")

    records = list(
        db[metadata["collection_name"]]
        .find({}, {"_id": 0, "row_index": 1, "values": 1})
        .sort("row_index", 1)
    )
    if not records:
        raise ValueError("The saved dataset does not contain any rows.")

    frame = pd.DataFrame([record["values"] for record in records])
    feature_names = list(metadata.get("feature_columns") or [])
    task = metadata.get("task")
    target_column = metadata.get("target_column")

    missing_features = [name for name in feature_names if name not in frame.columns]
    if missing_features:
        raise ValueError(
            "Feature columns were not found: " + ", ".join(missing_features)
        )

    if task == "Classification":
        if not target_column or target_column not in frame.columns:
            raise ValueError("The classification target column was not found.")
        labels = frame[target_column].astype(str)
        frame["target"] = labels
        frame["class_name"] = labels
        target_names = sorted(labels.dropna().unique().tolist())

    elif task == "Regression":
        if not target_column or target_column not in frame.columns:
            raise ValueError("The regression target column was not found.")
        numeric_target = pd.to_numeric(frame[target_column], errors="coerce")
        valid = numeric_target.notna()
        frame = frame.loc[valid].copy()
        frame["target"] = numeric_target.loc[valid].astype(float)
        target_names = None

    else:  # Clustering / unsupervised
        target_names = None

    # Restore useful numeric dtypes after the MongoDB round-trip.
    for column in feature_names:
        if column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            original_non_null = frame[column].notna().sum()
            if original_non_null and converted.notna().sum() == original_non_null:
                frame[column] = converted

    return frame.reset_index(drop=True), feature_names, target_names


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_")[:48]


def _safe_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # pandas converts NumPy scalars, NaN, timestamps and other values into
    # JSON-compatible Python values; MongoDB stores missing values as null.
    payload = frame.to_json(orient="records", date_format="iso")
    return json.loads(payload)


def save_custom_dataset(
    name: str,
    frame: pd.DataFrame,
    task: str,
    target_column: str | None,
    filename: str,
) -> str:
    """Persist dataset rows and metadata. Roll back its collection on failure."""
    db = get_database()
    if db is None:
        raise ConnectionError("MongoDB is not connected.")

    clean_name = str(name).strip()
    name_key = _slug(clean_name)
    if not name_key:
        raise ValueError("The dataset name is not valid.")
    if db["custom_datasets"].find_one({"name_key": name_key}):
        raise ValueError("This dataset name is already in use. Choose another name.")

    if task not in {"Classification", "Regression", "Clustering"}:
        raise ValueError("This analysis task is not supported.")

    stored_frame = frame.copy()
    if target_column and target_column in stored_frame.columns:
        feature_columns = [
            column for column in stored_frame.columns
            if column != target_column
        ]
    else:
        target_column = None
        feature_columns = list(stored_frame.columns)

    if not feature_columns:
        raise ValueError("The dataset must contain at least one feature.")

    numeric_feature_count = sum(
        pd.api.types.is_numeric_dtype(stored_frame[column])
        for column in feature_columns
    )
    if task in {"Regression", "Clustering"} and numeric_feature_count < (
        1 if task == "Regression" else 2
    ):
        raise ValueError(
            "There are not enough numeric features for the selected task."
        )

    records = _safe_json_records(stored_frame)
    dataset_id = uuid.uuid4().hex[:10]
    collection_name = f"custom_data_{name_key}_{dataset_id}"
    data_collection = db[collection_name]

    metadata = {
        "name": clean_name,
        "name_key": name_key,
        "dataset_id": dataset_id,
        "filename": str(filename),
        "collection_name": collection_name,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_feature_count": int(numeric_feature_count),
        "task": task,
        "row_count": int(len(stored_frame)),
        "column_count": int(len(stored_frame.columns)),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        for start in range(0, len(records), 1000):
            batch = records[start:start + 1000]
            data_collection.insert_many(
                [
                    {
                        "row_index": start + offset,
                        "values": record,
                    }
                    for offset, record in enumerate(batch)
                ],
                ordered=True,
            )

        db["custom_datasets"].insert_one(metadata)

        # Keep Home's latest-upload panel compatible with its existing source.
        db["dataset_uploads"].insert_one(
            {
                "filename": str(filename),
                "dataset_name": clean_name,
                "collection": collection_name,
                "rows": int(len(stored_frame)),
                "columns": list(stored_frame.columns),
                "kind": "custom_dataset",
                "created_at": metadata["created_at"],
            }
        )
    except Exception:
        # Do not leave an orphaned data collection if persistence fails.
        try:
            data_collection.drop()
        except Exception:
            pass
        try:
            db["custom_datasets"].delete_one({"dataset_id": dataset_id})
        except Exception:
            pass
        raise

    return clean_name


def render_dataset_upload_section(reserved_names: list[str] | None = None) -> None:
    """Render the additional CSV upload section at the bottom of Dataset Explorer."""
    reserved = {_slug(name) for name in (reserved_names or [])}
    catalog = get_custom_dataset_catalog()
    reserved.update(_slug(item.get("name", "")) for item in catalog)

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    st.subheader("Upload & Test a New Dataset")
    st.caption(
        "Upload your own CSV file. The dataset will be validated and saved to "
        "MongoDB, then made available in Dataset Explorer and Machine Learning Lab."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="custom_dataset_csv_upload",
        help="Maximum: 20 MB, 50,000 rows, and 100 columns.",
    )
    if uploaded is None:
        st.info("Choose a CSV file to preview the data and configure the dataset.")
        return

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("The file exceeds the 20 MB size limit.")
        return

    try:
        raw = uploaded.getvalue()
        sample_text = raw[:8192].decode("utf-8-sig", errors="replace")
        try:
            delimiter = csv.Sniffer().sniff(
                sample_text, delimiters=",;\\t|"
            ).delimiter
        except csv.Error:
            delimiter = ","
        try:
            preview = pd.read_csv(
                io.BytesIO(raw), sep=delimiter, encoding="utf-8-sig", low_memory=False
            )
        except UnicodeDecodeError:
            preview = pd.read_csv(
                io.BytesIO(raw), sep=delimiter, encoding="latin-1", low_memory=False
            )
    except Exception as exc:
        st.error(f"The CSV file could not be read: {exc}")
        return

    preview.columns = [str(column).strip() for column in preview.columns]
    preview = preview.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if preview.empty:
        st.error("The CSV contains no data after empty rows and columns are removed.")
        return
    if len(preview) < 10:
        st.error("The dataset must contain at least 10 rows.")
        return
    if len(preview) > MAX_ROWS:
        st.error("The dataset exceeds the 50,000-row limit.")
        return
    if len(preview.columns) > MAX_COLUMNS:
        st.error("The dataset exceeds the 100-column limit.")
        return
    if not preview.columns.is_unique or any(not column for column in preview.columns):
        st.error("Column names must be unique and cannot be empty.")
        return

    st.markdown("#### 1. Dataset details")
    name_col, task_col = st.columns([1.2, 1])
    with name_col:
        dataset_name = st.text_input(
            "Dataset name",
            value=os.path.splitext(uploaded.name)[0],
            key="custom_dataset_name_input",
        ).strip()
    with task_col:
        task = st.selectbox(
            "Analysis task",
            ["Classification", "Regression", "Clustering"],
            key="custom_dataset_task_input",
        )

    name_key = _slug(dataset_name)
    valid = True
    if not name_key:
        st.error("Enter a valid dataset name.")
        valid = False
    elif name_key in reserved:
        st.error("This dataset name is already registered. Choose another name.")
        valid = False

    target_column = None
    feature_columns = list(preview.columns)
    if task in {"Classification", "Regression"}:
        target_column = st.selectbox(
            "Target column",
            list(preview.columns),
            index=len(preview.columns) - 1,
            key="custom_dataset_target_input",
            help="The target column contains the label or value to predict.",
        )
        feature_columns = [
            column for column in preview.columns
            if column != target_column
        ]
        if not feature_columns:
            st.error("The dataset must contain at least one feature column.")
            valid = False

        target_values = preview[target_column]
        if target_values.isna().any():
            st.error(
                "The target column contains missing values. Fill them in or remove "
                "those rows from the CSV, then upload the file again."
            )
            valid = False

        if task == "Classification":
            if target_values.nunique(dropna=True) < 2:
                st.error("Classification requires at least two target classes.")
                valid = False
        else:
            numeric_target = pd.to_numeric(target_values, errors="coerce")
            numeric_target = numeric_target.replace([float("inf"), float("-inf")], pd.NA)
            if numeric_target.isna().any():
                st.error("The regression target must contain only valid numeric values (not missing or infinite).")
                valid = False
            else:
                preview[target_column] = numeric_target
            numeric_features = sum(
                pd.api.types.is_numeric_dtype(preview[column])
                for column in feature_columns
            )
            if numeric_features < 1:
                st.error("Regression requires at least one numeric feature.")
                valid = False
    else:
        numeric_features = sum(
            pd.api.types.is_numeric_dtype(preview[column])
            for column in feature_columns
        )
        if numeric_features < 2:
            st.error("Clustering requires at least two numeric features.")
            valid = False

    if task == "Classification":
        class_counts = target_values.value_counts(dropna=False)
        if len(class_counts) > 50:
            st.warning(
                "The target has more than 50 classes. Make sure this column "
                "is a categorical label."
            )

    # Duplicate feature names are blocked above. Limit extremely sparse CSVs.
    missing_percent = float(preview.isna().mean().mean() * 100)
    if missing_percent > 60:
        st.warning(
            f"{missing_percent:.1f}% of dataset values are missing. Review the data quality "
            "before saving the data or training a model."
        )

    st.markdown("#### 2. Preview and validation")
    stat_a, stat_b, stat_c = st.columns(3)
    stat_a.metric("Rows", f"{len(preview):,}")
    stat_b.metric("Columns", f"{len(preview.columns):,}")
    stat_c.metric("Missing values", f"{int(preview.isna().sum().sum()):,}")
    st.dataframe(preview.head(10), use_container_width=True, hide_index=True)
    st.caption(
        f"Analysis task: {task} · "
        + (f"Target: {target_column}" if target_column else "No target (unsupervised)")
        + f" · Features: {len(feature_columns)}"
    )

    db = get_database()
    if db is None:
        st.warning(
            "The preview is ready, but MongoDB is not connected. "
            "Connect MongoDB to save the dataset."
        )

    if st.button(
        "Validate & Save to MongoDB",
        type="primary",
        use_container_width=True,
        disabled=(not valid or db is None),
        key="custom_dataset_save_button",
    ):
        try:
            with st.spinner("Validating and saving the dataset to MongoDB..."):
                saved_name = save_custom_dataset(
                    dataset_name,
                    preview,
                    task,
                    target_column,
                    uploaded.name,
                )
            st.cache_data.clear()
            st.success(
                f"Dataset '{saved_name}' was saved successfully and is now available "
                "in Dataset Explorer and Machine Learning Lab."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"The dataset could not be saved: {exc}")
