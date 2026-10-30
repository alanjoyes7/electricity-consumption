import matplotlib.pyplot as plt
import streamlit as st

from kdd_pipeline import run_pipeline


st.set_page_config(
    page_title="Electricity Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Consumption Pattern Analysis")

st.markdown(
    """
    **S5 Data Mining Project**

    Discover electricity consumption patterns using
    data cleaning, preprocessing and K-Means clustering.
    """
)

st.sidebar.header("Configuration")

sample_size = st.sidebar.selectbox(
    "Number of records",
    [10000, 25000, 50000],
    index=2
)

selected_k = st.sidebar.selectbox(
    "Number of clusters",
    [2, 3, 4, 5, 6],
    index=1
)

if st.sidebar.button("Run Data Mining"):

    try:
        with st.spinner("Processing dataset..."):

            output = run_pipeline(
                sample_size=sample_size,
                selected_k=selected_k
            )

        st.session_state["output"] = output

    except Exception as error:
        st.error(f"Processing failed: {error}")


if "output" not in st.session_state:

    st.info(
        "Select the sample size and number of clusters "
        "in the sidebar, then click Run Data Mining."
    )

else:

    output = st.session_state["output"]

    raw = output["raw_data"]
    cleaned = output["cleaned_data"]
    results = output["results"]
    evaluation = output["evaluation"]
    metrics = output["metrics"]
    report = output["cleaning_report"]
    features = output["features"]

    # Dataset overview.
    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Records", f"{len(raw):,}")
    col2.metric("Cleaned Records", f"{len(cleaned):,}")
    col3.metric("Removed Records", f"{report['removed_rows']:,}")

    st.subheader("Original Data")
    st.dataframe(raw.head(10), use_container_width=True)

    st.subheader("Missing Values Before Cleaning")

    st.dataframe(
        raw.isnull().sum().rename("Missing Count").to_frame(),
        use_container_width=True
    )

    # Cleaned dataset.
    st.header("2. Cleaned Dataset")

    st.dataframe(
        cleaned.head(10),
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned CSV",
        data=cleaned.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # Model evaluation.
    st.header("3. K-Means Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected K", metrics["K"])

    col2.metric(
        "Silhouette Score",
        f"{metrics['Silhouette Score']:.4f}"
    )

    col3.metric(
        "Davies-Bouldin Index",
        f"{metrics['Davies-Bouldin Index']:.4f}"
    )

    # Elbow curve.
    st.subheader("Elbow Method")

    fig, ax = plt.subplots()

    ax.plot(
        evaluation["K"],
        evaluation["Inertia"],
        marker="o"
    )

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    plt.close(fig)

    # Silhouette comparison.
    st.subheader("Silhouette Score Comparison")

    fig, ax = plt.subplots()

    ax.plot(
        evaluation["K"],
        evaluation["Silhouette Score"],
        marker="o"
    )

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Quality")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Evaluation Table")

    st.dataframe(
        evaluation.round(4),
        use_container_width=True
    )

    # Cluster distribution.
    st.header("4. Cluster Distribution")

    distribution = (
        results["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(distribution)

    # PCA visualization.
    st.header("5. Cluster Visualization")

    fig, ax = plt.subplots()

    for cluster in sorted(results["Cluster"].unique()):

        subset = results[
            results["Cluster"] == cluster
        ]

        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            s=10,
            alpha=0.6,
            label=f"Cluster {cluster}"
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("K-Means Clusters Using PCA")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "PCA explained variance: "
        f"{output['explained_variance']:.2%}"
    )

    # Cluster profiles.
    st.header("6. Cluster Profiles")

    profiles = results.groupby("Cluster")[features].mean()

    st.dataframe(
        profiles.round(3),
        use_container_width=True
    )

    # Download final output.
    st.header("7. Export Results")

    st.download_button(
        "Download Clustered Dataset",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="clustered_data.csv",
        mime="text/csv"
    )