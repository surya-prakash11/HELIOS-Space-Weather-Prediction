"""
═══════════════════════════════════════════════════════════════
HELIOS - Exploratory Data Analysis Module
═══════════════════════════════════════════════════════════════
Comprehensive EDA with 6 major plot types:
1. Statistical summary
2. Correlation matrix
3. Distribution plots
4. Flare class analysis
5. Temporal analysis
6. Feature vs flare comparison
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    WAREHOUSE_PARQUET,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    COLOR_PALETTE_FLARES,
    PLOT_DPI,
    setup_logging,
)

logger = setup_logging("eda")

# Set plot style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ExploratoryAnalysis:
    """
    Performs comprehensive EDA on HELIOS warehouse data.

    Features:
    - Statistical summaries with skewness
    - Correlation analysis
    - Distribution visualizations
    - Temporal trend analysis
    - Flare class analysis
    - Feature comparison across classes

    Usage:
        eda = ExploratoryAnalysis()
        eda.run_all_analyses(df)
    """

    def __init__(self):
        """Initialize the EDA module."""
        self.viz_dir = VISUALIZATIONS_DIR
        self.reports_dir = REPORTS_DIR

        logger.info("Exploratory Analysis module initialized")

    def statistical_summary(self, df):
        """
        Generate comprehensive statistical summary.

        Args:
            df: Input DataFrame
        """
        logger.info("=" * 70)
        logger.info("1. STATISTICAL SUMMARY")
        logger.info("=" * 70)

        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe().T

        # Add missing percentage
        summary["missing_pct"] = (df[numeric_cols].isna().sum() / len(df) * 100).values

        # Add skewness
        summary["skewness"] = (
            df[numeric_cols].apply(lambda x: stats.skew(x.dropna())).values
        )

        # Add kurtosis
        summary["kurtosis"] = (
            df[numeric_cols].apply(lambda x: stats.kurtosis(x.dropna())).values
        )

        # Reorder columns
        col_order = [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "missing_pct",
            "skewness",
            "kurtosis",
        ]
        summary = summary[col_order]

        # Save to file
        summary_path = self.reports_dir / "statistical_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"✅ Statistical summary saved to: {summary_path}")

        # Print key statistics
        logger.info("\nKey Statistics (first 10 features):")
        print(summary.head(10).to_string())

        logger.info("\nHighly skewed features (|skewness| > 2):")
        high_skew = summary[abs(summary["skewness"]) > 2].sort_values(
            "skewness", ascending=False
        )
        if len(high_skew) > 0:
            print(high_skew[["mean", "std", "skewness"]].to_string())
        else:
            logger.info("  None found")

        return summary

    def correlation_matrix(self, df):
        """
        Create correlation matrix heatmap.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("2. CORRELATION ANALYSIS")
        logger.info("=" * 70)

        # Select key numeric features
        key_features = [
            "SW_Speed",
            "Proton_Density",
            "IMF_B_Avg",
            "Bz_GSE",
            "Flow_Pressure",
            "Kp_Index",
            "Dst_Index",
            "AE_Index",
            "flare_class_encoded",
        ]

        available_features = [f for f in key_features if f in df.columns]

        if len(available_features) < 2:
            logger.warning("Not enough features for correlation analysis")
            return

        # Calculate correlation
        corr_matrix = df[available_features].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
        )

        ax.set_title(
            "HELIOS - Feature Correlation Matrix",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        corr_path = self.viz_dir / "correlation_matrix.png"
        plt.savefig(corr_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Correlation matrix saved to: {corr_path}")
        plt.close()

        # Print top correlations with flare class
        if "flare_class_encoded" in corr_matrix.columns:
            flare_corr = (
                corr_matrix["flare_class_encoded"]
                .drop("flare_class_encoded")
                .sort_values(ascending=False)
            )
            logger.info("\nTop 10 features correlated with flare class:")
            print(flare_corr.head(10).to_string())

    def distribution_plots(self, df):
        """
        Create distribution plots for key features.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("3. FEATURE DISTRIBUTIONS")
        logger.info("=" * 70)

        # Select key features
        features = [
            "SW_Speed",
            "Proton_Density",
            "IMF_B_Avg",
            "Bz_GSE",
            "Flow_Pressure",
            "Temperature",
            "Kp_Index",
            "Dst_Index",
        ]

        available_features = [f for f in features if f in df.columns]

        if len(available_features) == 0:
            logger.warning("No features available for distribution plots")
            return

        # Create subplots
        n_features = len(available_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(available_features):
            ax = axes[idx]

            # Remove outliers for better visualization
            data = df[feature].dropna()
            q1 = data.quantile(0.01)
            q99 = data.quantile(0.99)
            data_filtered = data[(data >= q1) & (data <= q99)]

            # Histogram with KDE
            ax.hist(
                data_filtered,
                bins=50,
                alpha=0.6,
                color="skyblue",
                edgecolor="black",
                density=True,
            )

            # KDE overlay
            try:
                data_filtered.plot(kind="kde", ax=ax, color="darkblue", linewidth=2)
            except:
                pass

            # Mean and median lines
            mean_val = data_filtered.mean()
            median_val = data_filtered.median()

            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )

            ax.set_title(f"{feature} Distribution", fontweight="bold", fontsize=11)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(available_features), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "HELIOS - Feature Distributions", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        dist_path = self.viz_dir / "feature_distributions.png"
        plt.savefig(dist_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Distribution plots saved to: {dist_path}")
        plt.close()

    def flare_class_analysis(self, df):
        """
        Analyze and visualize flare class distribution.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("4. FLARE CLASS ANALYSIS")
        logger.info("=" * 70)

        if "flare_class" not in df.columns:
            logger.warning("No flare_class column found")
            return

        # Count flares by class
        class_counts = df["flare_class"].value_counts().sort_index()

        logger.info("\nFlare class distribution:")
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {cls:12s}: {count:8,} ({pct:6.2f}%)")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Bar chart (log scale)
        ax1 = axes[0]
        colors = [COLOR_PALETTE_FLARES.get(i, "gray") for i in range(len(class_counts))]
        bars = ax1.bar(
            range(len(class_counts)),
            class_counts.values,
            color=colors,
            edgecolor="black",
        )
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels(class_counts.index, rotation=45, ha="right")
        ax1.set_yscale("log")
        ax1.set_ylabel("Count (log scale)", fontweight="bold")
        ax1.set_title(
            "Flare Class Distribution (Log Scale)", fontweight="bold", fontsize=12
        )
        ax1.grid(axis="y", alpha=0.3)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 2: Pie chart (excluding No Flare)
        ax2 = axes[1]
        flare_only = class_counts.drop("No Flare", errors="ignore")

        if len(flare_only) > 0:
            colors_pie = [
                COLOR_PALETTE_FLARES.get(i + 1, "gray") for i in range(len(flare_only))
            ]
            wedges, texts, autotexts = ax2.pie(
                flare_only.values,
                labels=flare_only.index,
                autopct="%1.1f%%",
                colors=colors_pie,
                startangle=90,
                explode=[0.05] * len(flare_only),
            )

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax2.set_title(
                "Flare Events Distribution\n(Excluding No Flare)",
                fontweight="bold",
                fontsize=12,
            )

        # Plot 3: Stacked percentage bar
        ax3 = axes[2]
        percentages = (class_counts / len(df) * 100).values

        bottom = 0
        for i, (cls, pct) in enumerate(zip(class_counts.index, percentages)):
            color = COLOR_PALETTE_FLARES.get(i, "gray")
            ax3.barh(0, pct, left=bottom, color=color, edgecolor="black", label=cls)

            # Add label if percentage > 1%
            if pct > 1:
                ax3.text(
                    bottom + pct / 2,
                    0,
                    f"{cls}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )

            bottom += pct

        ax3.set_xlim(0, 100)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel("Percentage (%)", fontweight="bold")
        ax3.set_yticks([])
        ax3.set_title("Class Imbalance Visualization", fontweight="bold", fontsize=12)
        ax3.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        plt.suptitle(
            "HELIOS - Solar Flare Class Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        flare_path = self.viz_dir / "flare_class_distribution.png"
        plt.savefig(flare_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Flare class analysis saved to: {flare_path}")
        plt.close()

    def temporal_analysis(self, df):
        """
        Analyze temporal trends in solar wind and flare activity.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("5. TEMPORAL ANALYSIS")
        logger.info("=" * 70)

        if "timestamp" not in df.columns:
            logger.warning("No timestamp column found")
            return

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Resample to daily averages
        df_daily = (
            df.set_index("timestamp")
            .resample("D")
            .agg(
                {
                    "SW_Speed": "mean",
                    "Bz_GSE": "mean",
                    "Proton_Density": "mean",
                    "Dst_Index": "mean",
                }
            )
            .reset_index()
        )

        # Create 4-panel plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        features_to_plot = [
            ("SW_Speed", "Solar Wind Speed (km/s)", "blue"),
            ("Bz_GSE", "Bz GSE (nT)", "green"),
            ("Proton_Density", "Proton Density (n/cc)", "orange"),
            ("Dst_Index", "Dst Index (nT)", "red"),
        ]

        for idx, (feature, label, color) in enumerate(features_to_plot):
            ax = axes[idx]

            if feature not in df_daily.columns or df_daily[feature].isna().all():
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {feature}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot daily values
            ax.plot(
                df_daily["timestamp"],
                df_daily[feature],
                color=color,
                alpha=0.3,
                linewidth=0.5,
                label="Daily",
            )

            # Plot 30-day rolling average
            rolling_30d = df_daily[feature].rolling(window=30, min_periods=1).mean()
            ax.plot(
                df_daily["timestamp"],
                rolling_30d,
                color=color,
                linewidth=2,
                label="30-day avg",
            )

            # Mark major flare events
            if "flare_class_encoded" in df.columns:
                major_flares = df[df["flare_class_encoded"] >= 4]  # M and X class
                if len(major_flares) > 0:
                    ax.scatter(
                        major_flares["timestamp"],
                        [df_daily[feature].max() * 0.95] * len(major_flares),
                        color="red",
                        marker="v",
                        s=50,
                        alpha=0.6,
                        label="Major Flares (M/X)",
                        zorder=5,
                    )

            # Shade geomagnetic storm periods (Dst < -50)
            if feature == "Dst_Index":
                storm_periods = df_daily[df_daily["Dst_Index"] < -50]
                if len(storm_periods) > 0:
                    ax.axhspan(-500, -50, alpha=0.2, color="red", label="Storm Period")

            ax.set_ylabel(label, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

            # Add horizontal reference lines
            if feature == "Bz_GSE":
                ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            elif feature == "Dst_Index":
                ax.axhline(-50, color="orange", linestyle="--", linewidth=1, alpha=0.5)

        axes[-1].set_xlabel("Date", fontweight="bold")
        plt.suptitle(
            "HELIOS - Temporal Trends in Solar Wind Parameters",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        temporal_path = self.viz_dir / "temporal_analysis.png"
        plt.savefig(temporal_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Temporal analysis saved to: {temporal_path}")
        plt.close()

    def feature_vs_flare(self, df):
        """
        Compare feature distributions across flare classes.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("6. FEATURE vs FLARE CLASS COMPARISON")
        logger.info("=" * 70)

        if "flare_class" not in df.columns:
            logger.warning("No flare_class column found")
            return

        # Select features to compare
        features = ["SW_Speed", "Bz_GSE", "Proton_Density", "Kp_Index"]
        available_features = [f for f in features if f in df.columns]

        if len(available_features) == 0:
            logger.warning("No features available for comparison")
            return

        # Create box plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, feature in enumerate(available_features):
            ax = axes[idx]

            # Prepare data
            plot_data = (
                df[df["flare_class"] != "No Flare"].copy()
                if len(df) > 1000
                else df.copy()
            )

            # Create box plot
            sns.boxplot(
                data=plot_data,
                x="flare_class",
                y=feature,
                palette=COLOR_PALETTE_FLARES,
                ax=ax,
                showfliers=False,  # Hide outliers for clarity
            )

            ax.set_title(f"{feature} by Flare Class", fontweight="bold", fontsize=12)
            ax.set_xlabel("Flare Class", fontweight="bold")
            ax.set_ylabel(feature, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Rotate x labels
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "HELIOS - Solar Wind Conditions by Flare Class",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        feature_path = self.viz_dir / "feature_vs_flare.png"
        plt.savefig(feature_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Feature comparison saved to: {feature_path}")
        plt.close()

    def run_all_analyses(self, df):
        """
        Run all EDA analyses.

        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 70)

        # Run all analyses
        self.statistical_summary(df)
        self.correlation_matrix(df)
        self.distribution_plots(df)
        self.flare_class_analysis(df)
        self.temporal_analysis(df)
        self.feature_vs_flare(df)

        logger.info("\n" + "=" * 70)
        logger.info("EDA COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"All visualizations saved to: {self.viz_dir}")
        logger.info(f"All reports saved to: {self.reports_dir}")


def run_eda():
    """
    Convenience function to run EDA.
    """
    logger.info("Loading warehouse data...")

    if not WAREHOUSE_PARQUET.exists():
        raise FileNotFoundError(
            f"Warehouse data not found: {WAREHOUSE_PARQUET}\n"
            "Please run merger.py first!"
        )

    df = pd.read_parquet(WAREHOUSE_PARQUET)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    eda = ExploratoryAnalysis()
    eda.run_all_analyses(df)


if __name__ == "__main__":
    """
    Run EDA when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - EXPLORATORY DATA ANALYSIS")
    print("=" * 70 + "\n")

    try:
        run_eda()

        print("\n✅ SUCCESS!")
        print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        print(f"Reports saved to: {REPORTS_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("EDA failed")
        sys.exit(1)
