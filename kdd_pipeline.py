from pathlib import Path
import urllib.request
import zipfile
import shutil

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


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "household_power_consumption.txt"
ZIP_PATH = DATA_DIR / "household_power_consumption.zip"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. FEATURE SELECTION
# =========================================================

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


# =========================================================
# 3. AUTOMATIC DATASET DOWNLOAD
# =========================================================

def ensure_dataset():
    """
    Use the local dataset if available.
    Otherwise, download and extract it from UCI.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse an existing non-empty dataset.
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        print("Using existing electricity consumption dataset.")
        return DATA_PATH

    url = (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    )

    try:
        print("Downloading electricity consumption dataset...")

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with open(ZIP_PATH, "wb") as destination:
                shutil.copyfileobj(response, destination)

        # Validate the downloaded archive.
        if not zipfile.is_zipfile(ZIP_PATH):
            raise ValueError(
                "Downloaded file is not a valid ZIP archive."
            )

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:

            filename = next(
                (
                    name for name in archive.namelist()
                    if name.endswith(
                        "household_power_consumption.txt"
                    )
                ),
                None
            )

            if filename is None:
                raise FileNotFoundError(
                    "Dataset text file was not found inside the ZIP."
                )

            # Extract only the required dataset.
            with archive.open(filename) as source:
                with open(DATA_PATH, "wb") as destination:
                    shutil.copyfileobj(source, destination)

        if DATA_PATH.stat().st_size == 0:
            DATA_PATH.unlink(missing_ok=True)
            raise ValueError("The extracted dataset is empty.")

        print("Dataset downloaded and extracted successfully.")

        return DATA_PATH

    except Exception as error:
        DATA_PATH.unlink(missing_ok=True)

        raise RuntimeError(
            "Unable to download the electricity consumption dataset. "
            f"Reason: {error}"
        ) from error

    finally:
        ZIP_PATH.unlink(missing_ok=True)


# =========================================================
# 4. LOAD DATA
# =========================================================

def load_data(sample_size=50000):
    """
    Load electricity consumption records.
    Downloads the dataset automatically if it is missing.
    """

    data_path = ensure_dataset()

    df = pd.read_csv(
        data_path,
        sep=";",
        na_values=["?"],
        nrows=sample_size,
        low_memory=False
    )

    if df.empty:
        raise ValueError(
            "No records were loaded. Check the dataset file."
        )

    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    print(f"Loaded {len(df):,} records.")

    return df


# =========================================================
# 5. DATA CLEANING
# =========================================================

def clean_data(df):
    """Remove duplicates and invalid electricity measurements."""

    data = df.copy()
    original_rows = len(data)

    # Remove duplicate records.
    data = data.drop_duplicates().copy()

    # Convert selected features to numeric values.
    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Replace infinite values with missing values.
    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove records missing essential measurements.
    data = data.dropna(subset=FEATURES).copy()

    # Electricity consumption and sub-meter readings
    # should not contain negative values.
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

    # Voltage must be positive.
    data = data[data["Voltage"] > 0].copy()

    cleaning_report = {
        "original_rows": original_rows,
        "cleaned_rows": len(data),
        "removed_rows": original_rows - len(data)
    }

    if len(data) < 10:
        raise ValueError(
            "Too few records remain after data cleaning."
        )

    print("Data cleaning completed.")

    return data, cleaning_report


# =========================================================
# 6. DATA PREPROCESSING
# =========================================================

def preprocess_data(data):
    """
    Select features, impute missing values and standardize.
    """

    X = data[FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_imputed, X_scaled


# =========================================================
# 7. CLUSTER EVALUATION
# =========================================================

def evaluate_clusters(X_scaled):
    """
    Evaluate K-Means for K values from 2 to 6.
    """

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


# =========================================================
# 8. COMPLETE KDD PIPELINE
# =========================================================

def run_pipeline(sample_size=50000, selected_k=3):
    """
    Execute the complete Knowledge Discovery in Databases
    workflow for electricity consumption clustering.
    """

    # Step 1: Data collection and loading.
    raw_data = load_data(sample_size)

    # Step 2: Data cleaning.
    cleaned_data, cleaning_report = clean_data(raw_data)

    # Step 3: Feature selection and preprocessing.
    _, X_scaled = preprocess_data(cleaned_data)

    # Step 4: Evaluate different numbers of clusters.
    evaluation = evaluate_clusters(X_scaled)

    if selected_k not in evaluation["K"].tolist():
        raise ValueError(
            f"Invalid K={selected_k}. "
            f"Available values: {evaluation['K'].tolist()}"
        )

    # Step 5: Train the final K-Means model.
    model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # Step 6: Evaluate the final clustering.
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

    # Step 7: Reduce dimensions for visualization.
    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(X_scaled)

    results = cleaned_data.copy()

    results["Cluster"] = labels
    results["PCA1"] = coordinates[:, 0]
    results["PCA2"] = coordinates[:, 1]

    # Step 8: Save processed results.
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

    print("KDD pipeline completed successfully.")

    # Step 9: Return results to the dashboard.
    return {
        "raw_data": raw_data,
        "cleaned_data": cleaned_data,
        "results": results,
        "evaluation": evaluation,
        "cleaning_report": cleaning_report,
        "metrics": metrics,
        "features": FEATURES,
        "explained_variance": (
            pca.explained_variance_ratio_.sum()
        )
    }


# =========================================================
# 9. STANDALONE EXECUTION
# =========================================================

if __name__ == "__main__":

    output = run_pipeline(
        sample_size=50000,
        selected_k=3
    )

    print("\n========== CLEANING REPORT ==========")
    print(output["cleaning_report"])

    print("\n========== CLUSTER EVALUATION ==========")
    print(
        output["evaluation"]
        .round(4)
        .to_string(index=False)
    )

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