"""
═══════════════════════════════════════════════════════════════
HELIOS - Feature Engineering Module
═══════════════════════════════════════════════════════════════
Creates physics-based and temporal features for ML models
Features:
- Temporal: hour, month, day_of_week, season, solar_cycle
- Physics: solar_wind_energy, imf_total, bz_magnitude, temp_proxy
- Rolling windows: 3h, 6h, 24h means and stds
- Lag features: 1h, 3h, 6h, 12h, 24h
- Delta features: 1h and 6h changes
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    WAREHOUSE_PARQUET,
    ROLLING_WINDOWS,
    LAG_PERIODS,
    ROLLING_FEATURES,
    LAG_FEATURES,
    setup_logging,
)

logger = setup_logging("feature_engineer")


class FeatureEngineer:
    """
    Creates advanced features from solar wind data.

    Features created:
    1. Temporal features (time-based patterns)
    2. Physics features (domain knowledge)
    3. Rolling statistics (trends)
    4. Lag features (historical context)
    5. Delta features (rate of change)

    Usage:
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.rolling_windows = ROLLING_WINDOWS
        self.lag_periods = LAG_PERIODS
        self.rolling_features = ROLLING_FEATURES
        self.lag_features = LAG_FEATURES

        logger.info("Feature Engineer initialized")

    def create_temporal_features(self, df):
        """
        Create time-based features.

        Args:
            df: Input DataFrame with 'timestamp' column

        Returns:
            DataFrame with temporal features added
        """
        logger.info("Creating temporal features...")
        df = df.copy()

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Basic temporal features
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["quarter"] = df["timestamp"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Solar cycle phase (simplified: cycle 24 = 2008-2019, cycle 25 = 2019+)
        df["solar_cycle"] = np.where(df["year"] < 2020, 24, 25)

        # Season (Northern hemisphere)
        df["season"] = ((df["month"] % 12) // 3).astype(
            int
        )  # 0=Winter, 1=Spring, 2=Summer, 3=Fall

        logger.info(f"  Created {10} temporal features")

        return df

    def create_physics_features(self, df):
        """
        Create physics-based features using domain knowledge.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with physics features added
        """
        logger.info("Creating physics-based features...")
        df = df.copy()

        # Solar wind kinetic energy density
        # E = 0.5 * m * n * v^2 (simplified, m=proton mass factor absorbed)
        if "Proton_Density" in df.columns and "SW_Speed" in df.columns:
            df["solar_wind_energy"] = (
                0.5 * df["Proton_Density"] * (df["SW_Speed"] ** 2) / 1e6
            )

        # Total IMF magnitude
        if all(col in df.columns for col in ["Bx_GSE", "By_GSE", "Bz_GSE"]):
            df["imf_total"] = np.sqrt(
                df["Bx_GSE"] ** 2 + df["By_GSE"] ** 2 + df["Bz_GSE"] ** 2
            )

        # Bz magnitude and polarity
        if "Bz_GSE" in df.columns:
            df["bz_magnitude"] = np.abs(df["Bz_GSE"])
            df["bz_negative"] = (df["Bz_GSE"] < 0).astype(int)
            df["bz_southward_strength"] = np.where(
                df["Bz_GSE"] < 0, np.abs(df["Bz_GSE"]), 0
            )

        # Temperature proxy (from pressure and density)
        if "Flow_Pressure" in df.columns and "Proton_Density" in df.columns:
            df["temp_proxy"] = df["Flow_Pressure"] / (df["Proton_Density"] + 1e-10)

        # Dynamic pressure (ram pressure)
        if "Proton_Density" in df.columns and "SW_Speed" in df.columns:
            df["dynamic_pressure"] = (
                1.67e-6 * df["Proton_Density"] * (df["SW_Speed"] ** 2)
            )

        # Electric field (simplified: E = V × B)
        if "SW_Speed" in df.columns and "Bz_GSE" in df.columns:
            # Convert to mV/m: V(km/s) × B(nT) × 1e-3
            df["electric_field_calc"] = df["SW_Speed"] * np.abs(df["Bz_GSE"]) * 1e-3

        # Epsilon parameter (energy coupling function)
        # ε ∝ V² × B × sin⁴(θ/2), simplified for southward Bz
        if all(col in df.columns for col in ["SW_Speed", "Bz_GSE"]):
            df["epsilon"] = np.where(
                df["Bz_GSE"] < 0, (df["SW_Speed"] ** 2) * np.abs(df["Bz_GSE"]) / 1e6, 0
            )

        physics_features = [
            "solar_wind_energy",
            "imf_total",
            "bz_magnitude",
            "bz_negative",
            "bz_southward_strength",
            "temp_proxy",
            "dynamic_pressure",
            "electric_field_calc",
            "epsilon",
        ]

        created = [f for f in physics_features if f in df.columns]
        logger.info(f"  Created {len(created)} physics features")

        return df

    def create_rolling_features(self, df):
        """
        Create rolling window statistics (moving averages and stds).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with rolling features added
        """
        logger.info("Creating rolling window features...")
        df = df.copy()

        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        feature_count = 0

        for window in self.rolling_windows:
            for feature in self.rolling_features:
                if feature in df.columns:
                    # Rolling mean
                    col_name = f"{feature}_{window}h_mean"
                    df[col_name] = (
                        df[feature].rolling(window=window, min_periods=1).mean()
                    )
                    feature_count += 1

                    # Rolling std (only for larger windows)
                    if window >= 6:
                        col_name = f"{feature}_{window}h_std"
                        df[col_name] = (
                            df[feature].rolling(window=window, min_periods=1).std()
                        )
                        feature_count += 1

        logger.info(f"  Created {feature_count} rolling window features")

        return df

    def create_lag_features(self, df):
        """
        Create lag features (historical values).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with lag features added
        """
        logger.info("Creating lag features...")
        df = df.copy()

        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        feature_count = 0

        for lag in self.lag_periods:
            for feature in self.lag_features:
                if feature in df.columns:
                    col_name = f"{feature}_lag_{lag}h"
                    df[col_name] = df[feature].shift(lag)
                    feature_count += 1

        logger.info(f"  Created {feature_count} lag features")

        return df

    def create_delta_features(self, df):
        """
        Create delta features (rate of change).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with delta features added
        """
        logger.info("Creating delta (rate of change) features...")
        df = df.copy()

        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        delta_periods = [1, 6]  # 1-hour and 6-hour changes
        feature_count = 0

        for period in delta_periods:
            for feature in ["SW_Speed", "Bz_GSE", "IMF_B_Avg", "Proton_Density"]:
                if feature in df.columns:
                    col_name = f"{feature}_delta_{period}h"
                    df[col_name] = df[feature] - df[feature].shift(period)
                    feature_count += 1

        logger.info(f"  Created {feature_count} delta features")

        return df

    def create_interaction_features(self, df):
        """
        Create interaction features (products and ratios of key variables).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")
        df = df.copy()

        feature_count = 0

        # Speed × Density interaction
        if "SW_Speed" in df.columns and "Proton_Density" in df.columns:
            df["speed_density_product"] = df["SW_Speed"] * df["Proton_Density"]
            feature_count += 1

        # Bz × Speed interaction
        if "Bz_GSE" in df.columns and "SW_Speed" in df.columns:
            df["bz_speed_product"] = df["Bz_GSE"] * df["SW_Speed"]
            feature_count += 1

        # IMF / Speed ratio
        if "IMF_B_Avg" in df.columns and "SW_Speed" in df.columns:
            df["imf_speed_ratio"] = df["IMF_B_Avg"] / (df["SW_Speed"] + 1e-10)
            feature_count += 1

        logger.info(f"  Created {feature_count} interaction features")

        return df

    def create_all_features(self, df):
        """
        Create all engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all features added
        """
        logger.info("=" * 70)
        logger.info("STARTING FEATURE ENGINEERING PROCESS")
        logger.info("=" * 70)

        initial_cols = len(df.columns)

        # Create features in order
        df = self.create_temporal_features(df)
        df = self.create_physics_features(df)
        df = self.create_rolling_features(df)
        df = self.create_lag_features(df)
        df = self.create_delta_features(df)
        df = self.create_interaction_features(df)

        final_cols = len(df.columns)
        new_features = final_cols - initial_cols

        logger.info("\n" + "=" * 70)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Initial columns:  {initial_cols}")
        logger.info(f"Final columns:    {final_cols}")
        logger.info(f"New features:     {new_features}")
        logger.info(f"Total records:    {len(df):,}")
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return df


def engineer_features(df):
    """
    Convenience function to engineer features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    """
    Run feature engineering when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - FEATURE ENGINEERING")
    print("=" * 70 + "\n")

    try:
        from config import OMNI_PARQUET

        logger.info("Loading cleaned OMNI data...")
        df = pd.read_parquet(OMNI_PARQUET)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

        df_features = engineer_features(df)

        print("\n✅ SUCCESS!")
        print(
            f"Final dataset: {len(df_features):,} records × {len(df_features.columns)} columns"
        )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Feature engineering failed")
        sys.exit(1)
