"""
═══════════════════════════════════════════════════════════════
HELIOS - Data Cleaner Module
═══════════════════════════════════════════════════════════════
Handles missing values, outliers, and data quality issues
Strategies:
- Forward fill for physical parameters (max 6 hours)
- Linear interpolation for indices
- Zero-fill for proton flux
- Winsorization for outliers
- Drop rows with excessive missing data
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OMNI_PARQUET,
    VISUALIZATIONS_DIR,
    MAX_MISSING_PERCENT,
    OUTLIER_PERCENTILES,
    MAX_FORWARD_FILL_HOURS,
    VALID_RANGES,
    setup_logging,
    PLOT_DPI,
)

logger = setup_logging("data_cleaner")


class OMNIDataCleaner:
    """
    Cleans OMNI solar wind data.

    Features:
    - Multiple imputation strategies by variable type
    - Outlier detection and handling
    - Data quality reporting
    - Missing value visualization

    Usage:
        cleaner = OMNIDataCleaner()
        df_clean = cleaner.clean(df)
    """

    def __init__(self):
        """Initialize the data cleaner."""
        self.max_missing_pct = MAX_MISSING_PERCENT
        self.outlier_percentiles = OUTLIER_PERCENTILES
        self.max_ffill_hours = MAX_FORWARD_FILL_HOURS
        self.valid_ranges = VALID_RANGES

        logger.info("OMNI Data Cleaner initialized")

    def analyze_missing(self, df):
        """
        Analyze missing data patterns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing data statistics
        """
        logger.info("Analyzing missing data patterns...")

        missing_stats = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing_Count": df.isna().sum().values,
                "Missing_Percent": (df.isna().sum() / len(df) * 100).values,
                "Non_Missing_Count": df.notna().sum().values,
            }
        )

        missing_stats = missing_stats.sort_values("Missing_Percent", ascending=False)

        logger.info("\nMissing Data Summary:")
        logger.info(missing_stats.to_string(index=False))

        return missing_stats

    def visualize_missing(self, df, save_path=None):
        """
        Create visualization of missing data.

        Args:
            df: Input DataFrame
            save_path: Path to save plot
        """
        logger.info("Creating missing data visualization...")

        # Calculate missing percentages
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=True)
        missing_pct = missing_pct[
            missing_pct > 0
        ]  # Only show columns with missing data

        if len(missing_pct) == 0:
            logger.info("No missing data to visualize!")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing_pct) * 0.3)))

        colors = [
            "#e74c3c" if x > 50 else "#f39c12" if x > 20 else "#3498db"
            for x in missing_pct.values
        ]

        bars = ax.barh(range(len(missing_pct)), missing_pct.values, color=colors)
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index)
        ax.set_xlabel("Missing Data (%)", fontsize=12)
        ax.set_title(
            "OMNI Data - Missing Value Analysis", fontsize=14, fontweight="bold"
        )
        ax.axvline(
            x=20, color="orange", linestyle="--", alpha=0.5, label="20% threshold"
        )
        ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, missing_pct.values)):
            ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = VISUALIZATIONS_DIR / "missing_values_report.png"

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"Missing data plot saved: {save_path}")
        plt.close()

    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate strategies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        df = df.copy()

        # Strategy 1: Forward fill for physical parameters (limited to 6 hours)
        physical_params = [
            "SW_Speed",
            "Proton_Density",
            "Temperature",
            "Flow_Pressure",
            "IMF_B_Avg",
            "Bx_GSE",
            "By_GSE",
            "Bz_GSE",
        ]

        for col in physical_params:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                df[col] = df[col].fillna(method="ffill", limit=self.max_ffill_hours)
                missing_after = df[col].isna().sum()
                filled = missing_before - missing_after
                logger.info(f"  {col:20s}: Forward filled {filled:,} values")

        # Strategy 2: Linear interpolation for indices
        index_params = ["Kp_Index", "Dst_Index", "AE_Index", "R_Sunspot"]

        for col in index_params:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                missing_after = df[col].isna().sum()
                filled = missing_before - missing_after
                logger.info(f"  {col:20s}: Interpolated {filled:,} values")

        # Strategy 3: Zero-fill for proton flux (no flux = 0)
        flux_params = [col for col in df.columns if "Proton_Flux" in col]

        for col in flux_params:
            missing_before = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            filled = missing_before
            logger.info(f"  {col:20s}: Zero-filled {filled:,} values")

        # Strategy 4: Drop rows with excessive missing data (>60% columns missing)
        initial_rows = len(df)
        missing_per_row = df.isna().sum(axis=1)
        threshold = len(df.columns) * (self.max_missing_pct / 100)
        df = df[missing_per_row <= threshold]
        rows_dropped = initial_rows - len(df)
        logger.info(
            f"\nDropped {rows_dropped:,} rows with >{self.max_missing_pct}% missing data"
        )

        return df

    def detect_outliers(self, df, column):
        """
        Detect outliers using percentile method.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            Boolean mask of outliers
        """
        if column not in df.columns or df[column].isna().all():
            return pd.Series([False] * len(df), index=df.index)

        lower_percentile = np.percentile(
            df[column].dropna(), self.outlier_percentiles[0]
        )
        upper_percentile = np.percentile(
            df[column].dropna(), self.outlier_percentiles[1]
        )

        outliers = (df[column] < lower_percentile) | (df[column] > upper_percentile)

        return outliers

    def handle_outliers(self, df):
        """
        Handle outliers using winsorization (clipping to percentiles).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        logger.info("\nHandling outliers using winsorization...")
        df = df.copy()

        # Columns to check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_check = [
            col for col in numeric_cols if col not in ["Year", "DOY", "Hour"]
        ]

        for col in cols_to_check:
            if df[col].isna().all():
                continue

            outliers = self.detect_outliers(df, col)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                # Winsorize (clip to percentiles)
                lower = df[col].quantile(self.outlier_percentiles[0] / 100)
                upper = df[col].quantile(self.outlier_percentiles[1] / 100)

                df[col] = df[col].clip(lower=lower, upper=upper)

                logger.info(
                    f"  {col:20s}: Clipped {n_outliers:,} outliers to [{lower:.2f}, {upper:.2f}]"
                )

        return df

    def validate_physical_ranges(self, df):
        """
        Validate data against known physical parameter ranges.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with validation report
        """
        logger.info("\nValidating physical parameter ranges...")

        validation_report = []

        for col, (min_val, max_val) in self.valid_ranges.items():
            if col in df.columns:
                below_min = (df[col] < min_val).sum()
                above_max = (df[col] > max_val).sum()
                valid = len(df) - below_min - above_max - df[col].isna().sum()

                validation_report.append(
                    {
                        "Parameter": col,
                        "Expected_Range": f"[{min_val}, {max_val}]",
                        "Valid_Count": valid,
                        "Below_Min": below_min,
                        "Above_Max": above_max,
                        "Valid_Percent": (valid / len(df) * 100),
                    }
                )

                if below_min + above_max > 0:
                    logger.warning(
                        f"  {col}: {below_min} below min, {above_max} above max"
                    )

        validation_df = pd.DataFrame(validation_report)
        return validation_df

    def clean(self, df, visualize=True):
        """
        Complete cleaning pipeline.

        Args:
            df: Input DataFrame
            visualize: Whether to create visualizations

        Returns:
            Cleaned DataFrame
        """
        logger.info("=" * 70)
        logger.info("STARTING DATA CLEANING PROCESS")
        logger.info("=" * 70)

        initial_rows = len(df)
        initial_cols = len(df.columns)

        # Step 1: Analyze missing data
        missing_stats = self.analyze_missing(df)

        if visualize:
            self.visualize_missing(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Handle outliers
        df = self.handle_outliers(df)

        # Step 4: Validate ranges
        validation_report = self.validate_physical_ranges(df)

        # Final summary
        final_rows = len(df)
        final_cols = len(df.columns)

        logger.info("\n" + "=" * 70)
        logger.info("DATA CLEANING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Initial records: {initial_rows:,}")
        logger.info(f"Final records:   {final_rows:,}")
        logger.info(
            f"Records removed: {initial_rows - final_rows:,} ({(initial_rows-final_rows)/initial_rows*100:.2f}%)"
        )
        logger.info(f"Columns:         {final_cols}")

        remaining_missing = df.isna().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        logger.info(
            f"\nRemaining missing values: {remaining_missing:,} ({remaining_missing/total_values*100:.2f}%)"
        )

        logger.info("=" * 70)
        logger.info("DATA CLEANING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return df


def clean_omni_data():
    """
    Convenience function to load and clean OMNI data.

    Returns:
        Cleaned DataFrame
    """
    logger.info("Loading OMNI data from file...")

    if not OMNI_PARQUET.exists():
        raise FileNotFoundError(
            f"OMNI data file not found: {OMNI_PARQUET}\n"
            "Please run data_loader.py first!"
        )

    df = pd.read_parquet(OMNI_PARQUET)
    logger.info(f"Loaded {len(df):,} records")

    cleaner = OMNIDataCleaner()
    df_clean = cleaner.clean(df, visualize=True)

    # Save cleaned data
    logger.info(f"\nSaving cleaned data to {OMNI_PARQUET}")
    df_clean.to_parquet(OMNI_PARQUET, index=False)
    df_clean.to_csv(OMNI_CSV, index=False)
    logger.info("✅ Cleaned data saved!")

    return df_clean


if __name__ == "__main__":
    """
    Run data cleaning when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - OMNI DATA CLEANER")
    print("=" * 70 + "\n")

    try:
        df_clean = clean_omni_data()

        print("\n✅ SUCCESS!")
        print(f"Cleaned dataset: {len(df_clean):,} records")
        print(f"Files saved to: {PROCESSED_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Data cleaning failed")
        sys.exit(1)
