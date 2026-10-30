from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "household_power_consumption.txt"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


def load_data(sample_size=50000):
    """Load electricity consumption measurements."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if DATA_PATH.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values=["?"],
        nrows=sample_size,
        low_memory=False
    )

    if df.empty:
        raise ValueError(
            "No records found. Check the dataset contents."
        )

    return df


def clean_data(df):
    """Clean invalid values and duplicate records."""

    data = df.copy()

    original_rows = len(data)

    data = data.drop_duplicates().copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove records missing essential measurements.
    data = data.dropna(subset=FEATURES).copy()

    # Remove invalid negative measurements.
    nonnegative_features = [
        "Global_active_power",
        "Global_reactive_power",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for column in nonnegative_features:
        data = data[data[column] >= 0]

    data = data[data["Voltage"] > 0].copy()

    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(data),
        "removed_rows": original_rows - len(data)
    }

    if len(data) < 10:
        raise ValueError(
            "Too few records remain after cleaning."
        )

    return data, report


def preprocess_data(data):
    """Impute missing values and standardize features."""

    X = data[FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_imputed, X_scaled


def evaluate_clusters(X_scaled):
    """Evaluate K-Means for K values from 2 to 6."""

    evaluation_results = []

    max_k = min(6, len(X_scaled) - 1)

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(5000, len(X_scaled)),
            random_state=42
        )

        db_index = davies_bouldin_score(
            X_scaled,
            labels
        )

        evaluation_results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": score,
            "Davies-Bouldin Index": db_index
        })

    return pd.DataFrame(evaluation_results)


def run_pipeline(sample_size=50000, selected_k=3):
    """Execute the complete KDD workflow."""

    # 1. Load data.
    raw_data = load_data(sample_size)

    # 2. Clean data.
    cleaned_data, cleaning_report = clean_data(raw_data)

    # 3. Preprocess data.
    _, X_scaled = preprocess_data(cleaned_data)

    # 4. Evaluate different cluster counts.
    evaluation = evaluate_clusters(X_scaled)

    if selected_k not in evaluation["K"].tolist():
        raise ValueError(
            f"Choose K from {evaluation['K'].tolist()}"
        )

    # 5. Train final K-Means model.
    model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # 6. Evaluate final model.
    metrics = {
        "K": selected_k,
        "Silhouette Score": silhouette_score(
            X_scaled,
            labels,
            sample_size=min(5000, len(X_scaled)),
            random_state=42
        ),
        "Davies-Bouldin Index": davies_bouldin_score(
            X_scaled,
            labels
        ),
        "Inertia": model.inertia_
    }

    # 7. Prepare visualization coordinates.
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    results = cleaned_data.copy()

    results["Cluster"] = labels
    results["PCA1"] = coordinates[:, 0]
    results["PCA2"] = coordinates[:, 1]

    # 8. Save output files.
    cleaned_data.to_csv(
        OUTPUT_DIR / "cleaned_data.csv",
        index=False
    )

    results.to_csv(
        OUTPUT_DIR / "clustered_data.csv",
        index=False
    )

    evaluation.to_csv(
        OUTPUT_DIR / "cluster_evaluation.csv",
        index=False
    )

    return {
        "raw_data": raw_data,
        "cleaned_data": cleaned_data,
        "results": results,
        "evaluation": evaluation,
        "cleaning_report": cleaning_report,
        "metrics": metrics,
        "features": FEATURES,
        "explained_variance": pca.explained_variance_ratio_.sum()
    }


if __name__ == "__main__":

    output = run_pipeline(
        sample_size=50000,
        selected_k=3
    )

    print("\n========== CLEANING REPORT ==========")
    print(output["cleaning_report"])

    print("\n========== CLUSTER EVALUATION ==========")
    print(output["evaluation"].round(4).to_string(index=False))

    print("\n========== FINAL MODEL METRICS ==========")
    print(output["metrics"])

    print("\n========== CLUSTER DISTRIBUTION ==========")
    print(
        output["results"]["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\n========== CLUSTERED DATA PREVIEW ==========")
    print(
        output["results"]
        .head(10)
        .to_string(index=False)
    )

    print("\nOutput files saved to:", OUTPUT_DIR)