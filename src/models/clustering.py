"""
═══════════════════════════════════════════════════════════════
HELIOS - K-Means Clustering Module
═══════════════════════════════════════════════════════════════
Discovers natural solar wind patterns using unsupervised learning

Features:
- Optimal k selection (Elbow + Silhouette + Davies-Bouldin)
- K-Means clustering on scaled features
- PCA visualization (2D projection)
- Cluster profiling and interpretation
- Flare rate analysis by cluster
- Radar charts for cluster characteristics

Usage:
    clusterer = SolarWindClusterer()
    clusterer.find_optimal_k()
    clusterer.train(k=3)
    clusterer.analyze()
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from config import (
    WAREHOUSE_PARQUET,
    KMEANS_PATH,
    SCALER_PATH,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    KMEANS_PARAMS,
    CLUSTER_NAMES,
    CLUSTER_COLORS,
    PLOT_DPI,
    setup_logging,
)

logger = setup_logging("clustering")


class SolarWindClusterer:
    """
    K-Means clustering for solar wind pattern discovery.

    Features:
    - Automatic optimal k detection
    - Cluster profiling
    - PCA visualization
    - Flare occurrence analysis per cluster

    Usage:
        clusterer = SolarWindClusterer()
        optimal_k = clusterer.find_optimal_k()
        clusterer.train(k=optimal_k)
        clusterer.analyze()
    """

    def __init__(self):
        """Initialize the clusterer."""
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.feature_names = None
        self.cluster_names = CLUSTER_NAMES
        self.cluster_colors = CLUSTER_COLORS

        logger.info("Solar Wind Clusterer initialized")

    def load_data(self):
        """
        Load warehouse data for clustering.

        Returns:
            DataFrame with features
        """
        logger.info("Loading warehouse data for clustering...")

        if not WAREHOUSE_PARQUET.exists():
            raise FileNotFoundError(
                f"Warehouse data not found: {WAREHOUSE_PARQUET}\n"
                "Please run the ETL pipeline first!"
            )

        df = pd.read_parquet(WAREHOUSE_PARQUET)
        logger.info(f"Loaded {len(df):,} records")

        return df

    def select_clustering_features(self, df):
        """
        Select relevant features for clustering.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        # Key physical parameters for clustering
        clustering_features = [
            "SW_Speed",
            "Proton_Density",
            "IMF_B_Avg",
            "Bz_GSE",
            "Flow_Pressure",
            "Temperature",
            "Kp_Index",
            "Dst_Index",
            "AE_Index",
        ]

        # Add physics features if available
        physics_features = [
            "solar_wind_energy",
            "bz_magnitude",
            "imf_total",
            "dynamic_pressure",
            "temp_proxy",
        ]

        available_features = [f for f in clustering_features if f in df.columns]
        available_physics = [f for f in physics_features if f in df.columns]

        all_features = available_features + available_physics

        if len(all_features) == 0:
            raise ValueError("No clustering features found in data!")

        logger.info(f"Selected {len(all_features)} features for clustering")
        self.feature_names = all_features

        # Extract features and handle missing values
        X = df[all_features].copy()

        # Fill missing with median
        X = X.fillna(X.median())

        # Remove any remaining NaN or inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        logger.info(f"Feature matrix shape: {X.shape}")

        return X

    def find_optimal_k(self, X=None, k_range=range(2, 11)):
        """
        Find optimal number of clusters using multiple metrics.

        Args:
            X: Feature matrix (optional, will load if None)
            k_range: Range of k values to test

        Returns:
            Optimal k value
        """
        logger.info("=" * 70)
        logger.info("FINDING OPTIMAL NUMBER OF CLUSTERS")
        logger.info("=" * 70)

        if X is None:
            df = self.load_data()
            X = self.select_clustering_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Test different k values
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []

        logger.info(f"Testing k values: {list(k_range)}")

        for k in k_range:
            logger.info(f"\nTesting k={k}...")

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            inertias.append(inertia)

            # Silhouette score (higher is better, range: -1 to 1)
            sil_score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(sil_score)

            # Davies-Bouldin index (lower is better)
            db_score = davies_bouldin_score(X_scaled, labels)
            davies_bouldin_scores.append(db_score)

            logger.info(f"  Inertia: {inertia:.2f}")
            logger.info(f"  Silhouette: {sil_score:.4f}")
            logger.info(f"  Davies-Bouldin: {db_score:.4f}")

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Elbow method
        ax1 = axes[0]
        ax1.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Clusters (k)", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Inertia", fontweight="bold", fontsize=12)
        ax1.set_title("Elbow Method", fontweight="bold", fontsize=14)
        ax1.grid(alpha=0.3)

        # Plot 2: Silhouette score
        ax2 = axes[1]
        ax2.plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Clusters (k)", fontweight="bold", fontsize=12)
        ax2.set_ylabel("Silhouette Score", fontweight="bold", fontsize=12)
        ax2.set_title(
            "Silhouette Analysis (Higher is Better)", fontweight="bold", fontsize=14
        )
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

        # Plot 3: Davies-Bouldin index
        ax3 = axes[2]
        ax3.plot(k_range, davies_bouldin_scores, "ro-", linewidth=2, markersize=8)
        ax3.set_xlabel("Number of Clusters (k)", fontweight="bold", fontsize=12)
        ax3.set_ylabel("Davies-Bouldin Index", fontweight="bold", fontsize=12)
        ax3.set_title(
            "Davies-Bouldin Index (Lower is Better)", fontweight="bold", fontsize=14
        )
        ax3.grid(alpha=0.3)

        plt.suptitle("K-Means Cluster Optimization", fontsize=16, fontweight="bold")
        plt.tight_layout()

        elbow_path = VISUALIZATIONS_DIR / "elbow_method.png"
        plt.savefig(elbow_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"\n✅ Elbow plot saved to: {elbow_path}")
        plt.close()

        # Determine optimal k (max silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMAL K SELECTION")
        logger.info("=" * 70)
        logger.info(f"Optimal k (by Silhouette): {optimal_k}")
        logger.info(f"Best Silhouette Score: {max(silhouette_scores):.4f}")
        logger.info("=" * 70)

        return optimal_k

    def train(self, k=None, X=None):
        """
        Train K-Means model.

        Args:
            k: Number of clusters (if None, uses config default)
            X: Feature matrix (optional, will load if None)
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING K-MEANS CLUSTERING MODEL")
        logger.info("=" * 70)

        if k is None:
            k = KMEANS_PARAMS["n_clusters"]

        if X is None:
            df = self.load_data()
            X = self.select_clustering_features(df)

        # Scale features
        logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Train K-Means
        logger.info(f"Training K-Means with k={k}...")
        self.model = KMeans(
            n_clusters=k,
            n_init=KMEANS_PARAMS["n_init"],
            max_iter=KMEANS_PARAMS["max_iter"],
            random_state=KMEANS_PARAMS["random_state"],
        )

        labels = self.model.fit_predict(X_scaled)

        # Evaluate
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)

        logger.info(f"\n✅ K-Means training completed")
        logger.info(f"Number of clusters: {k}")
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            pct = count / len(labels) * 100
            logger.info(f"  Cluster {cluster_id}: {count:,} samples ({pct:.2f}%)")

        logger.info("=" * 70)

        return labels

    def analyze(self, df=None):
        """
        Analyze and visualize clusters.

        Args:
            df: Input DataFrame (optional, will load if None)
        """
        logger.info("\n" + "=" * 70)
        logger.info("ANALYZING CLUSTER CHARACTERISTICS")
        logger.info("=" * 70)

        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        if df is None:
            df = self.load_data()

        X = self.select_clustering_features(df)
        X_scaled = self.scaler.transform(X)

        # Get cluster labels
        labels = self.model.predict(X_scaled)
        df["cluster"] = labels

        # Cluster profiles (mean values)
        logger.info("\nCluster Profiles (Mean Values):")
        cluster_profiles = df.groupby("cluster")[self.feature_names].mean()
        print(cluster_profiles.to_string())

        # Save cluster profiles
        profile_path = REPORTS_DIR / "cluster_profiles.csv"
        cluster_profiles.to_csv(profile_path)
        logger.info(f"\n✅ Cluster profiles saved to: {profile_path}")

        # Visualizations
        self.plot_pca_clusters(X_scaled, labels)
        self.plot_cluster_radar(cluster_profiles)
        self.plot_flare_rate_by_cluster(df)

        logger.info("=" * 70)

    def plot_pca_clusters(self, X_scaled, labels):
        """
        Plot clusters in 2D using PCA.

        Args:
            X_scaled: Scaled feature matrix
            labels: Cluster labels
        """
        logger.info("\nCreating PCA visualization...")

        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each cluster
        unique_labels = np.unique(labels)
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            color = (
                list(self.cluster_colors.values())[cluster_id]
                if cluster_id < len(self.cluster_colors)
                else "gray"
            )
            label = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")

            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=color,
                label=label,
                alpha=0.6,
                s=30,
                edgecolors="black",
                linewidths=0.5,
            )

        # Plot cluster centers
        centers_scaled = self.model.cluster_centers_
        centers_pca = self.pca.transform(centers_scaled)

        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c="black",
            marker="X",
            s=300,
            edgecolors="white",
            linewidths=2,
            label="Centroids",
            zorder=10,
        )

        ax.set_xlabel(
            f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_ylabel(
            f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_title(
            "Solar Wind Clusters - PCA Visualization", fontweight="bold", fontsize=14
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        pca_path = VISUALIZATIONS_DIR / "kmeans_clusters_pca.png"
        plt.savefig(pca_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ PCA plot saved to: {pca_path}")
        plt.close()

    def plot_cluster_radar(self, cluster_profiles):
        """
        Create radar chart of cluster profiles.

        Args:
            cluster_profiles: DataFrame with cluster means
        """
        logger.info("Creating cluster radar chart...")

        # Normalize profiles to 0-1 scale for radar chart
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        profiles_norm = pd.DataFrame(
            scaler.fit_transform(cluster_profiles.T).T,
            columns=cluster_profiles.columns,
            index=cluster_profiles.index,
        )

        # Select top features for radar (max 8 for readability)
        top_features = profiles_norm.std().nlargest(8).index.tolist()
        profiles_radar = profiles_norm[top_features]

        # Number of variables
        num_vars = len(top_features)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Plot each cluster
        for idx, cluster_id in enumerate(profiles_radar.index):
            values = profiles_radar.loc[cluster_id].tolist()
            values += values[:1]  # Complete the circle

            color = (
                list(self.cluster_colors.values())[cluster_id]
                if cluster_id < len(self.cluster_colors)
                else "gray"
            )
            label = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")

            ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_features, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
        ax.grid(True)

        ax.set_title(
            "Cluster Profiles - Radar Chart\n(Normalized Values)",
            fontweight="bold",
            fontsize=14,
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        radar_path = VISUALIZATIONS_DIR / "cluster_profiles_radar.png"
        plt.savefig(radar_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Radar chart saved to: {radar_path}")
        plt.close()

    def plot_flare_rate_by_cluster(self, df):
        """
        Plot flare occurrence rate by cluster.

        Args:
            df: DataFrame with cluster assignments and flare data
        """
        logger.info("Analyzing flare rates by cluster...")

        if "flare_class_encoded" not in df.columns:
            logger.warning("No flare data available for analysis")
            return

        # Calculate flare rate by cluster
        flare_rates = (
            df.groupby("cluster")
            .agg({"flare_class_encoded": lambda x: (x > 0).sum() / len(x) * 100})
            .reset_index()
        )
        flare_rates.columns = ["cluster", "flare_rate_pct"]

        # Count major flares (M and X class)
        major_flare_counts = (
            df[df["flare_class_encoded"] >= 4].groupby("cluster").size()
        )
        flare_rates["major_flares"] = (
            flare_rates["cluster"].map(major_flare_counts).fillna(0)
        )

        logger.info("\nFlare rates by cluster:")
        print(flare_rates.to_string(index=False))

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Overall flare rate
        ax1 = axes[0]
        colors_list = [
            list(self.cluster_colors.values())[i] for i in flare_rates["cluster"]
        ]
        bars = ax1.bar(
            flare_rates["cluster"],
            flare_rates["flare_rate_pct"],
            color=colors_list,
            edgecolor="black",
            linewidth=1.5,
        )

        ax1.set_xlabel("Cluster", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Flare Rate (%)", fontweight="bold", fontsize=12)
        ax1.set_title(
            "Flare Occurrence Rate by Cluster", fontweight="bold", fontsize=14
        )
        ax1.set_xticks(flare_rates["cluster"])
        ax1.set_xticklabels(
            [self.cluster_names.get(i, f"Cluster {i}") for i in flare_rates["cluster"]],
            rotation=15,
            ha="right",
        )
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Plot 2: Major flare counts
        ax2 = axes[1]
        bars2 = ax2.bar(
            flare_rates["cluster"],
            flare_rates["major_flares"],
            color=colors_list,
            edgecolor="black",
            linewidth=1.5,
        )

        ax2.set_xlabel("Cluster", fontweight="bold", fontsize=12)
        ax2.set_ylabel("Count", fontweight="bold", fontsize=12)
        ax2.set_title(
            "Major Flares (M+X Class) by Cluster", fontweight="bold", fontsize=14
        )
        ax2.set_xticks(flare_rates["cluster"])
        ax2.set_xticklabels(
            [self.cluster_names.get(i, f"Cluster {i}") for i in flare_rates["cluster"]],
            rotation=15,
            ha="right",
        )
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        plt.suptitle(
            "Flare Activity Analysis by Solar Wind Cluster",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        flare_path = VISUALIZATIONS_DIR / "flare_rate_by_cluster.png"
        plt.savefig(flare_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Flare rate plot saved to: {flare_path}")
        plt.close()

    def save_model(self):
        """Save trained model and scaler."""
        logger.info("\nSaving clustering model...")

        if self.model is None:
            logger.warning("No model to save!")
            return

        joblib.dump(self.model, KMEANS_PATH)
        logger.info(f"✅ K-Means model saved to: {KMEANS_PATH}")

        # Note: Scaler is shared with classifier, so we don't overwrite it here
        logger.info("✅ Model saved successfully")


def run_clustering():
    """
    Convenience function to run complete clustering analysis.
    """
    clusterer = SolarWindClusterer()

    # Load data
    df = clusterer.load_data()
    X = clusterer.select_clustering_features(df)

    # Find optimal k
    optimal_k = clusterer.find_optimal_k(X)

    # Train with optimal k
    clusterer.train(k=optimal_k, X=X)

    # Analyze clusters
    clusterer.analyze(df)

    # Save model
    clusterer.save_model()

    return clusterer


if __name__ == "__main__":
    """
    Run clustering when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - K-MEANS CLUSTERING")
    print("=" * 70 + "\n")

    try:
        clusterer = run_clustering()

        print("\n✅ SUCCESS!")
        print(f"Model saved to: {KMEANS_PATH}")
        print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        print(f"Reports saved to: {REPORTS_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Clustering failed")
        sys.exit(1)
