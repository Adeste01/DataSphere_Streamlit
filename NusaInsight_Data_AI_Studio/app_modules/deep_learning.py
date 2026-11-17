"""Lightweight PyTorch tabular deep-learning workflow for NusaInsight."""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _read_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name.upper(), default) or "").strip()


@__import__("functools").lru_cache(maxsize=2)
def _mongo_client(uri: str):
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    client.admin.command("ping")
    return client


def _get_database():
    uri = _read_secret("mongo_uri") or _read_secret("MONGODB_URI")
    if not uri:
        return None
    db_name = _read_secret("mongo_database") or _read_secret("MONGODB_DATABASE", "datasphere")
    return _mongo_client(uri)[db_name]


def _task_datasets(dataset_tasks: dict[str, str], custom_catalog: list[dict[str, Any]], task: str) -> list[str]:
    options = [
        name for name, task_label in dataset_tasks.items()
        if task.lower() in str(task_label).lower()
    ]
    options.extend(
        item["name"] for item in custom_catalog
        if item.get("task") == task and item.get("name") not in options
    )
    return options


def _prepare_arrays(frame: pd.DataFrame, feature_names: list[str], task: str, test_size: float, seed: int):
    if len(frame) < 20:
        raise ValueError("At least 20 valid rows are required for a train/test split.")

    target_name = "class_name" if task == "Classification" and "class_name" in frame.columns else "target"
    if target_name not in frame.columns:
        raise ValueError("The selected dataset does not contain a target column.")

    target = frame[target_name].copy()
    usable = target.notna()
    features = frame.loc[usable, feature_names].copy()
    target = target.loc[usable]

    if task == "Classification":
        labels = target.astype(str)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        classes = encoder.classes_.tolist()
        if len(classes) < 2:
            raise ValueError("Classification requires at least two target classes.")
        class_counts = np.unique(y, return_counts=True)[1]
        test_rows = int(np.ceil(test_size * len(y)))
        train_rows = len(y) - test_rows
        if class_counts.min() < 2 or test_rows < len(classes) or train_rows < len(classes):
            raise ValueError(
                "Each class needs at least two rows, and the train/test split must contain enough rows for every class. "
                "Add more samples or increase the test size."
            )
        stratify = y
    else:
        y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        features = features.loc[valid]
        y = y[valid]
        classes = None
        stratify = None
        if len(y) < 20:
            raise ValueError("At least 20 valid numeric target values are required for regression.")

    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify
    )
    x_train_raw = features.iloc[train_idx].replace([np.inf, -np.inf, "?"], np.nan)
    x_test_raw = features.iloc[test_idx].replace([np.inf, -np.inf, "?"], np.nan)

    x_train = pd.get_dummies(x_train_raw, drop_first=False, dtype=float)
    x_test = pd.get_dummies(x_test_raw, drop_first=False, dtype=float)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    x_train = x_train.apply(pd.to_numeric, errors="coerce")
    x_test = x_test.apply(pd.to_numeric, errors="coerce")

    medians = x_train.median().replace([np.inf, -np.inf], np.nan).fillna(0)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    y_train = np.asarray(y)[train_idx]
    y_test = np.asarray(y)[test_idx]
    return x_train_scaled, x_test_scaled, y_train, y_test, classes, x_train.columns.tolist(), scaler


def _train_mlp(x_train, x_test, y_train, y_test, task: str, epochs: int, seed: int):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install the project requirements first.") from exc

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    input_size = int(x_train.shape[1])
    output_size = int(np.max(y_train)) + 1 if task == "Classification" else 1
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_size),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss() if task == "Classification" else nn.MSELoss()

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    if task == "Classification":
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    history: list[float] = []
    batch_size = min(64, len(x_train_tensor))
    model.train()
    for _epoch in range(int(epochs)):
        permutation = torch.randperm(len(x_train_tensor))
        batch_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            prediction = model(x_train_tensor[indices])
            loss = criterion(prediction, y_train_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        history.append(float(np.mean(batch_losses)))

    model.eval()
    with torch.no_grad():
        raw_prediction = model(x_test_tensor).cpu().numpy()

    if task == "Classification":
        prediction = raw_prediction.argmax(axis=1)
        metrics = {
            "Accuracy": float(accuracy_score(y_test, prediction)),
            "Precision": float(precision_score(y_test, prediction, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, prediction, average="weighted", zero_division=0)),
            "F1-score": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
        }
    else:
        prediction = raw_prediction.reshape(-1)
        mse = mean_squared_error(y_test, prediction)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, prediction)),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_test, prediction)),
        }

    return model, metrics, history, prediction


def _save_deep_learning_run(model, db, metadata: dict[str, Any]) -> str | None:
    if db is None:
        return None
    import torch
    from gridfs import GridFS

    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": metadata["feature_count"],
            "architecture": [64, 32],
            "task": metadata["task"],
            "dataset": metadata["dataset"],
            "model_name": "PyTorch MLP",
            "feature_names": metadata["feature_names"],
            "classes": metadata.get("classes"),
        },
        buffer,
    )
    buffer.seek(0)
    fs = GridFS(db, collection="model_artifacts")
    file_id = fs.put(
        buffer.getvalue(),
        filename=f"deep_learning_{metadata['dataset']}_{metadata['created_at'].strftime('%Y%m%d_%H%M%S')}.pt",
        content_type="application/octet-stream",
        metadata={
            "task": metadata["task"],
            "dataset": metadata["dataset"],
            "model_name": "PyTorch MLP",
            "framework": "PyTorch",
            "created_at": metadata["created_at"],
        },
    )
    log = {
        "created_at": metadata["created_at"],
        "task": metadata["task"],
        "dataset": metadata["dataset"],
        "model_name": "PyTorch MLP",
        "framework": "PyTorch",
        "epochs": metadata["epochs"],
        "sample_count": metadata["sample_count"],
        "feature_count": metadata["feature_count"],
        "train_rows": metadata["train_rows"],
        "test_rows": metadata["test_rows"],
        "metrics": metadata["metrics"],
        "artifact_file_id": str(file_id),
        "status": "completed",
        "source": "deep_learning",
    }
    db["training_logs"].insert_one(log)
    return str(file_id)


def render_deep_learning(load_dataset, dataset_tasks: dict[str, str]) -> None:
    import streamlit as st

    st.markdown('<div class="ds-eyebrow">NUSAINSIGHT • NEURAL NETWORKS</div>', unsafe_allow_html=True)
    st.title("Deep Learning")
    st.markdown(
        '<div class="ds-subtitle">Train a lightweight PyTorch neural network for tabular classification or regression.</div>',
        unsafe_allow_html=True,
    )

    try:
        from custom_dataset_manager import get_custom_dataset_catalog
        custom_catalog = get_custom_dataset_catalog()
    except Exception:
        custom_catalog = []

    task = st.selectbox(
        "Learning task",
        ["Classification", "Regression"],
        key="deep_learning_task",
    )
    datasets = _task_datasets(dataset_tasks, custom_catalog, task)
    if not datasets:
        st.info(f"No {task.lower()} datasets are available yet. Add a dataset in Dataset Explorer.")
        return

    left, right = st.columns([0.9, 1.4], gap="large")
    with left:
        dataset_name = st.selectbox("Dataset", datasets, key=f"deep_learning_dataset_{task}")
        test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05, key="deep_learning_test_size")
        epochs = st.slider("Training epochs", 10, 100, 30, 5, key="deep_learning_epochs")
        seed = int(st.number_input("Random seed", 0, 999999, 42, key="deep_learning_seed"))
        st.caption("Small CPU model: 64 → 32 hidden units, ReLU activation, Adam optimizer.")
        run = st.button("Train neural network", type="primary", use_container_width=True, key="run_deep_learning")

    with right:
        st.markdown("**Model workflow**")
        st.write("1. Split the data into training and test sets.")
        st.write("2. Encode categorical inputs and scale numerical features.")
        st.write("3. Train a compact neural network on the CPU.")
        st.write("4. Evaluate predictions on held-out test data.")
        st.write("5. Save the model artifact and training log to MongoDB when connected.")

    if not run:
        st.info("Choose a dataset and settings, then train the neural network.")
        return

    try:
        with st.spinner("Preparing data and training the PyTorch model..."):
            frame, feature_names, _ = load_dataset(dataset_name)
            prepared = _prepare_arrays(frame, feature_names, task, float(test_size), seed)
            x_train, x_test, y_train, y_test, classes, prepared_features, scaler = prepared
            model, metrics, history, predictions = _train_mlp(
                x_train, x_test, y_train, y_test, task, int(epochs), seed
            )

        st.success("Neural network training completed.")
        metric_cols = st.columns(len(metrics))
        for col, (label, value) in zip(metric_cols, metrics.items()):
            col.metric(label, f"{value:.4f}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.plot(range(1, len(history) + 1), history, color="#0F766E", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("Training loss by epoch")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if task == "Classification":
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            cm = confusion_matrix(y_test, predictions)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted class")
            ax_cm.set_ylabel("Actual class")
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

        db = None
        try:
            db = _get_database()
            if db is not None:
                created_at = datetime.now(timezone.utc)
                metadata = {
                    "created_at": created_at,
                    "task": task,
                    "dataset": dataset_name,
                    "epochs": int(epochs),
                    "sample_count": int(len(frame)),
                    "feature_count": int(len(prepared_features)),
                    "train_rows": int(len(x_train)),
                    "test_rows": int(len(x_test)),
                    "feature_names": prepared_features,
                    "classes": classes,
                    "metrics": metrics,
                }
                artifact_id = _save_deep_learning_run(model, db, metadata)
                st.success(f"Model artifact and training log saved to MongoDB. Artifact ID: {artifact_id}")
            else:
                st.warning("Training completed, but MongoDB is not connected. The result was not saved permanently.")
        except Exception as exc:
            st.warning(f"Training completed, but the MongoDB save failed: {type(exc).__name__}.")
    except Exception as exc:
        st.error(f"Deep learning failed: {exc}")
