"""
═══════════════════════════════════════════════════════════════
HELIOS - Data Merger Module
═══════════════════════════════════════════════════════════════
Merges OMNI solar wind data with solar flare event data
Creates final data warehouse with Star Schema
Performs OLAP operations demonstration
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OMNI_PARQUET,
    FLARE_RAW,
    WAREHOUSE_PARQUET,
    WAREHOUSE_CSV,
    FLARE_CLASS_ENCODING,
    REPORTS_DIR,
    setup_logging,
)

logger = setup_logging("merger")


class DataMerger:
    """
    Merges OMNI and Solar Flare datasets.

    Features:
    - Auto-detects flare CSV column names
    - Handles multiple flare formats
    - Rounds timestamps to nearest hour
    - Keeps most severe flare per hour
    - LEFT JOIN to preserve all OMNI records
    - Encodes flare classes
    - Creates Star Schema warehouse

    Usage:
        merger = DataMerger()
        warehouse_df = merger.merge_datasets()
    """

    def __init__(self):
        """Initialize the data merger."""
        self.flare_encoding = FLARE_CLASS_ENCODING

        logger.info("Data Merger initialized")

    def load_omni_data(self):
        """
        Load cleaned OMNI data.

        Returns:
            OMNI DataFrame
        """
        logger.info("Loading OMNI data...")

        if not OMNI_PARQUET.exists():
            raise FileNotFoundError(
                f"OMNI data not found: {OMNI_PARQUET}\n"
                "Please run data_loader.py and data_cleaner.py first!"
            )

        df = pd.read_parquet(OMNI_PARQUET)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        logger.info(f"  Loaded {len(df):,} OMNI records")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def detect_flare_columns(self, df):
        """
        Auto-detect column names in flare CSV.

        Args:
            df: Flare DataFrame

        Returns:
            Dict with detected column names
        """
        columns = df.columns.str.lower()

        # Detect timestamp column
        timestamp_col = None
        for pattern in [
            "timestamp",
            "datetime",
            "date",
            "time",
            "start_time",
            "peak_time",
        ]:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                timestamp_col = matches[0]
                break

        # Detect class column
        class_col = None
        for pattern in ["class", "flare_class", "type", "category"]:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                class_col = matches[0]
                break

        # Detect peak flux column
        flux_col = None
        for pattern in ["flux", "peak", "intensity", "magnitude"]:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                flux_col = matches[0]
                break

        # Detect duration column
        duration_col = None
        for pattern in ["duration", "length", "time"]:
            matches = [
                col
                for col in df.columns
                if pattern in col.lower() and "start" not in col.lower()
            ]
            if matches:
                duration_col = matches[0]
                break

        detected = {
            "timestamp": timestamp_col,
            "class": class_col,
            "flux": flux_col,
            "duration": duration_col,
        }

        logger.info("  Detected flare columns:")
        for key, val in detected.items():
            logger.info(f"    {key:12s}: {val}")

        return detected

    def load_flare_data(self):
        """
        Load solar flare event data.

        Returns:
            Flare DataFrame or None if file doesn't exist
        """
        logger.info("Loading solar flare data...")

        if not FLARE_RAW.exists():
            logger.warning(f"Flare data not found: {FLARE_RAW}")
            logger.warning(
                "Will proceed with OMNI-only mode (all records marked as No Flare)"
            )
            return None

        try:
            # Try reading CSV
            df = pd.read_csv(FLARE_RAW)
            logger.info(f"  Loaded {len(df):,} flare records")

            # Auto-detect columns
            cols = self.detect_flare_columns(df)

            if cols["timestamp"] is None:
                logger.error("Could not detect timestamp column in flare data!")
                return None

            # Rename columns to standard names
            rename_map = {}
            if cols["timestamp"]:
                rename_map[cols["timestamp"]] = "timestamp"
            if cols["class"]:
                rename_map[cols["class"]] = "flare_class"
            if cols["flux"]:
                rename_map[cols["flux"]] = "peak_flux"
            if cols["duration"]:
                rename_map[cols["duration"]] = "duration"

            df = df.rename(columns=rename_map)

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Remove invalid timestamps
            invalid = df["timestamp"].isna().sum()
            if invalid > 0:
                logger.warning(
                    f"  Removed {invalid} flare records with invalid timestamps"
                )
                df = df.dropna(subset=["timestamp"])

            # Extract flare class letter (A, B, C, M, X)
            if "flare_class" in df.columns:
                df["flare_class_letter"] = (
                    df["flare_class"].astype(str).str[0].str.upper()
                )

                # Validate classes
                valid_classes = set("ABCMX")
                df["flare_class_letter"] = df["flare_class_letter"].apply(
                    lambda x: x if x in valid_classes else None
                )

            logger.info(
                f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
            )

            if "flare_class_letter" in df.columns:
                class_counts = df["flare_class_letter"].value_counts().sort_index()
                logger.info("  Flare class distribution:")
                for cls, count in class_counts.items():
                    logger.info(f"    {cls}: {count:,}")

            return df

        except Exception as e:
            logger.error(f"Error loading flare data: {e}")
            return None

    def round_to_hour(self, df, timestamp_col="timestamp"):
        """
        Round timestamps to nearest hour.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with rounded timestamps
        """
        df = df.copy()
        df[timestamp_col] = df[timestamp_col].dt.round("H")
        return df

    def aggregate_flares_per_hour(self, flare_df):
        """
        Aggregate multiple flares per hour, keeping most severe.

        Args:
            flare_df: Flare DataFrame

        Returns:
            Aggregated DataFrame with one row per hour
        """
        logger.info("Aggregating flares by hour...")

        # Define severity order
        severity_order = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}

        if "flare_class_letter" not in flare_df.columns:
            logger.warning("No flare_class_letter column found")
            return flare_df

        # Map to severity
        flare_df["severity"] = flare_df["flare_class_letter"].map(severity_order)

        # Group by hour and keep most severe
        agg_dict = {
            "flare_class_letter": lambda x: (
                x.iloc[x.map(severity_order).idxmax()] if len(x) > 0 else None
            ),
            "severity": "max",
        }

        if "peak_flux" in flare_df.columns:
            agg_dict["peak_flux"] = "max"
        if "duration" in flare_df.columns:
            agg_dict["duration"] = "sum"

        flare_agg = flare_df.groupby("timestamp").agg(agg_dict).reset_index()

        duplicates_removed = len(flare_df) - len(flare_agg)
        logger.info(f"  Aggregated {duplicates_removed:,} duplicate hourly flares")
        logger.info(f"  Final flare records: {len(flare_agg):,}")

        return flare_agg

    def merge_datasets(self, omni_df=None, flare_df=None):
        """
        Merge OMNI and flare datasets.

        Args:
            omni_df: OMNI DataFrame (optional, will load if None)
            flare_df: Flare DataFrame (optional, will load if None)

        Returns:
            Merged warehouse DataFrame
        """
        logger.info("=" * 70)
        logger.info("STARTING DATA MERGE PROCESS")
        logger.info("=" * 70)

        # Load data if not provided
        if omni_df is None:
            omni_df = self.load_omni_data()

        if flare_df is None:
            flare_df = self.load_flare_data()

        # Round OMNI timestamps to hour
        omni_df = self.round_to_hour(omni_df)

        if flare_df is not None and len(flare_df) > 0:
            # Round flare timestamps to hour
            flare_df = self.round_to_hour(flare_df)

            # Aggregate flares per hour
            flare_df = self.aggregate_flares_per_hour(flare_df)

            # Perform LEFT JOIN
            logger.info("Performing LEFT JOIN on timestamp...")
            merged_df = omni_df.merge(
                flare_df[["timestamp", "flare_class_letter", "peak_flux", "duration"]],
                on="timestamp",
                how="left",
            )

            flares_matched = merged_df["flare_class_letter"].notna().sum()
            logger.info(f"  Matched {flares_matched:,} flare events to OMNI records")

        else:
            logger.info("No flare data available - proceeding in OMNI-only mode")
            merged_df = omni_df.copy()
            merged_df["flare_class_letter"] = None
            merged_df["peak_flux"] = None
            merged_df["duration"] = None

        # Fill missing flare classes with 'No Flare'
        merged_df["flare_class"] = merged_df["flare_class_letter"].fillna("No Flare")

        # Encode flare classes
        logger.info("Encoding flare classes...")
        merged_df["flare_class_encoded"] = (
            merged_df["flare_class"].map(self.flare_encoding).fillna(0).astype(int)
        )

        # Add has_flare flag
        merged_df["has_flare"] = (merged_df["flare_class_encoded"] > 0).astype(int)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("DATA MERGE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total records:        {len(merged_df):,}")
        logger.info(
            f"Date range:           {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}"
        )
        logger.info(f"Flare events:         {merged_df['has_flare'].sum():,}")
        logger.info(f"No-flare periods:     {(merged_df['has_flare']==0).sum():,}")

        logger.info("\nFlare class distribution:")
        class_dist = merged_df["flare_class"].value_counts().sort_index()
        for cls, count in class_dist.items():
            pct = count / len(merged_df) * 100
            logger.info(f"  {cls:12s}: {count:8,} ({pct:5.2f}%)")

        logger.info("=" * 70)
        logger.info("DATA MERGE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return merged_df

    def create_star_schema(self, df):
        """
        Create Star Schema data warehouse tables.

        Args:
            df: Merged DataFrame

        Returns:
            Dict with fact table and dimension tables
        """
        logger.info("\nCreating Star Schema data warehouse...")

        # FACT TABLE: solar_events_fact
        fact_cols = [
            "timestamp",
            "flare_class_encoded",
            "SW_Speed",
            "Bz_GSE",
            "IMF_B_Avg",
            "Proton_Density",
            "Flow_Pressure",
            "Kp_Index",
            "Dst_Index",
            "AE_Index",
            "peak_flux",
            "duration",
        ]

        available_cols = [col for col in fact_cols if col in df.columns]
        fact_table = df[available_cols].copy()
        fact_table.insert(0, "event_id", range(1, len(fact_table) + 1))

        # DIMENSION 1: time_dim
        time_dim = pd.DataFrame(
            {
                "time_id": range(1, len(df) + 1),
                "timestamp": df["timestamp"],
                "year": df["timestamp"].dt.year,
                "month": df["timestamp"].dt.month,
                "day": df["timestamp"].dt.day,
                "hour": df["timestamp"].dt.hour,
                "day_of_week": df["timestamp"].dt.dayofweek,
                "quarter": df["timestamp"].dt.quarter,
                "is_weekend": (df["timestamp"].dt.dayofweek >= 5).astype(int),
                "season": ((df["timestamp"].dt.month % 12) // 3).astype(int),
            }
        )

        # DIMENSION 2: flare_class_dim
        flare_class_dim = pd.DataFrame(
            {
                "class_id": [0, 1, 2, 3, 4, 5],
                "class_name": ["No Flare", "A", "B", "C", "M", "X"],
                "severity_level": [
                    "None",
                    "Minimal",
                    "Low",
                    "Moderate",
                    "High",
                    "Extreme",
                ],
                "description": [
                    "Normal solar conditions",
                    "Background level",
                    "Minor flare",
                    "Small flare with minor effects",
                    "Medium flare with radio blackouts",
                    "Major flare with widespread impacts",
                ],
            }
        )

        # DIMENSION 3: solar_wind_dim
        def categorize_speed(speed):
            if pd.isna(speed):
                return "Unknown"
            elif speed < 350:
                return "Slow"
            elif speed < 500:
                return "Moderate"
            else:
                return "Fast"

        def categorize_density(density):
            if pd.isna(density):
                return "Unknown"
            elif density < 8:
                return "Low"
            elif density < 16:
                return "Medium"
            else:
                return "High"

        def categorize_bz(bz):
            if pd.isna(bz):
                return "Unknown"
            elif bz < -2:
                return "Southward"
            elif bz <= 2:
                return "Neutral"
            else:
                return "Northward"

        solar_wind_dim = pd.DataFrame(
            {
                "sw_id": range(1, len(df) + 1),
                "speed_category": (
                    df["SW_Speed"].apply(categorize_speed)
                    if "SW_Speed" in df.columns
                    else "Unknown"
                ),
                "density_category": (
                    df["Proton_Density"].apply(categorize_density)
                    if "Proton_Density" in df.columns
                    else "Unknown"
                ),
                "bz_polarity": (
                    df["Bz_GSE"].apply(categorize_bz)
                    if "Bz_GSE" in df.columns
                    else "Unknown"
                ),
            }
        )

        schema = {
            "fact_table": fact_table,
            "time_dim": time_dim,
            "flare_class_dim": flare_class_dim,
            "solar_wind_dim": solar_wind_dim,
        }

        logger.info("  Created Star Schema tables:")
        logger.info(
            f"    Fact table:        {len(fact_table):,} rows × {len(fact_table.columns)} cols"
        )
        logger.info(
            f"    Time dimension:    {len(time_dim):,} rows × {len(time_dim.columns)} cols"
        )
        logger.info(
            f"    Flare dimension:   {len(flare_class_dim):,} rows × {len(flare_class_dim.columns)} cols"
        )
        logger.info(
            f"    Solar wind dim:    {len(solar_wind_dim):,} rows × {len(solar_wind_dim.columns)} cols"
        )

        return schema

    def demonstrate_olap_operations(self, df):
        """
        Demonstrate OLAP operations on the warehouse.

        Args:
            df: Warehouse DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("DEMONSTRATING OLAP OPERATIONS")
        logger.info("=" * 70)

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # 1. ROLL-UP: Aggregate from monthly to yearly
        logger.info("\n1. ROLL-UP: Monthly flare counts → Yearly totals")
        monthly_counts = df[df["has_flare"] == 1].groupby(["year", "month"]).size()
        yearly_rollup = monthly_counts.groupby("year").sum()
        logger.info(yearly_rollup.to_string())

        # 2. DRILL-DOWN: Yearly → Monthly → Daily
        logger.info("\n2. DRILL-DOWN: Year 2015 → Monthly breakdown")
        if 2015 in df["year"].values:
            monthly_2015 = (
                df[(df["year"] == 2015) & (df["has_flare"] == 1)]
                .groupby("month")
                .size()
            )
            logger.info(monthly_2015.to_string())

        # 3. SLICE: Single dimension (year = 2020)
        logger.info("\n3. SLICE: Only year 2020 data")
        if 2020 in df["year"].values:
            slice_2020 = df[df["year"] == 2020]
            logger.info(f"   Total records in 2020: {len(slice_2020):,}")
            logger.info(f"   Flare events in 2020: {slice_2020['has_flare'].sum():,}")

        # 4. DICE: Multiple dimensions (M+X flares with high speed)
        logger.info("\n4. DICE: M+X class flares with SW_Speed > 500 km/s")
        if "SW_Speed" in df.columns:
            dice_result = df[(df["flare_class_encoded"] >= 4) & (df["SW_Speed"] > 500)]
            logger.info(f"   Matching records: {len(dice_result):,}")

        # 5. PIVOT: Flare counts by class and year
        logger.info("\n5. PIVOT: Flare counts by class × year")
        pivot_table = pd.pivot_table(
            df[df["has_flare"] == 1],
            values="has_flare",
            index="flare_class",
            columns="year",
            aggfunc="count",
            fill_value=0,
        )
        logger.info(pivot_table.to_string())

        # Save OLAP results
        olap_report_path = REPORTS_DIR / "olap_operations.txt"
        with open(olap_report_path, "w") as f:
            f.write("HELIOS - OLAP OPERATIONS DEMONSTRATION\n")
            f.write("=" * 70 + "\n\n")
            f.write("1. ROLL-UP: Yearly flare totals\n")
            f.write(yearly_rollup.to_string() + "\n\n")
            f.write("5. PIVOT: Flare counts by class × year\n")
            f.write(pivot_table.to_string() + "\n")

        logger.info(f"\n✅ OLAP report saved to: {olap_report_path}")
        logger.info("=" * 70)


def merge_and_create_warehouse():
    """
    Convenience function to merge data and create warehouse.

    Returns:
        Warehouse DataFrame
    """
    merger = DataMerger()

    # Merge datasets
    warehouse_df = merger.merge_datasets()

    # Create Star Schema
    star_schema = merger.create_star_schema(warehouse_df)

    # Demonstrate OLAP
    merger.demonstrate_olap_operations(warehouse_df)

    # Save warehouse
    logger.info(f"\nSaving data warehouse to {WAREHOUSE_PARQUET}")
    warehouse_df.to_parquet(WAREHOUSE_PARQUET, index=False)
    warehouse_df.to_csv(WAREHOUSE_CSV, index=False)

    # Save Star Schema tables
    for table_name, table_df in star_schema.items():
        table_path = PROCESSED_DIR / f"{table_name}.csv"
        table_df.to_csv(table_path, index=False)
        logger.info(f"  Saved {table_name} to {table_path}")

    logger.info("✅ Data warehouse created and saved!")

    return warehouse_df


if __name__ == "__main__":
    """
    Run data merge when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - DATA MERGER & WAREHOUSE BUILDER")
    print("=" * 70 + "\n")

    try:
        from config import PROCESSED_DIR

        warehouse_df = merge_and_create_warehouse()

        print("\n✅ SUCCESS!")
        print(f"Warehouse created: {len(warehouse_df):,} records")
        print(f"Files saved to: {PROCESSED_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Data merge failed")
        sys.exit(1)
