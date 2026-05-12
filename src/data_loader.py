"""
═══════════════════════════════════════════════════════════════
HELIOS - Data Loader Module
═══════════════════════════════════════════════════════════════
Loads and processes OMNI2 solar wind data files (.dat format)
Handles multiple years (2010-2025)
Converts Day-of-Year format to proper timestamps
Replaces fill values with NaN
Saves consolidated dataset
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    OMNI_COLUMNS,
    OMNI_FILL_VALUES,
    OMNI_KEEP_COLS,
    OMNI_PARQUET,
    OMNI_CSV,
    setup_logging,
)

logger = setup_logging("data_loader")


class OMNIDataLoader:
    """
    Loads OMNI2 solar wind data from fixed-width .dat files.

    Features:
    - Reads multiple yearly files
    - Handles Day-of-Year (DOY) to datetime conversion
    - Replaces fill values with NaN
    - Removes duplicates
    - Saves as Parquet and CSV

    Usage:
        loader = OMNIDataLoader()
        df = loader.load_all_years()
    """

    def __init__(self):
        """Initialize the OMNI data loader."""
        self.raw_dir = RAW_DIR
        self.processed_dir = PROCESSED_DIR
        self.columns = OMNI_COLUMNS
        self.fill_values = OMNI_FILL_VALUES
        self.keep_cols = OMNI_KEEP_COLS

        logger.info("OMNI Data Loader initialized")
        logger.info(f"Raw data directory: {self.raw_dir}")
        logger.info(f"Processed data directory: {self.processed_dir}")

    def find_omni_files(self):
        """
        Find all OMNI2 .dat files in raw directory.

        Returns:
            List of Path objects for OMNI files
        """
        omni_files = sorted(self.raw_dir.glob("omni2_*.dat"))

        if not omni_files:
            logger.error(f"No OMNI2 files found in {self.raw_dir}")
            logger.error("Expected files like: omni2_2010.dat, omni2_2011.dat, etc.")
            raise FileNotFoundError(
                f"No OMNI2 data files found in {self.raw_dir}\n"
                "Please download OMNI2 files from NASA OMNIWeb:\n"
                "https://omniweb.gsfc.nasa.gov/form/dx1.html"
            )

        logger.info(f"Found {len(omni_files)} OMNI2 files")
        for f in omni_files:
            logger.info(f"  - {f.name}")

        return omni_files

    def doy_to_datetime(self, year, doy, hour):
        """
        Convert Year, Day-of-Year, and Hour to datetime.

        Args:
            year: Year (int)
            doy: Day of year (1-366)
            hour: Hour of day (0-23)

        Returns:
            datetime object
        """
        try:
            base_date = datetime(int(year), 1, 1)
            target_date = base_date + timedelta(days=int(doy) - 1, hours=int(hour))
            return target_date
        except Exception as e:
            logger.warning(f"Error converting DOY to datetime: {e}")
            return pd.NaT

    def load_single_file(self, filepath):
        """
        Load a single OMNI2 .dat file.

        Args:
            filepath: Path to .dat file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading file: {filepath.name}")

        try:
            # Read fixed-width space-delimited file
            df = pd.read_csv(
                filepath,
                sep=r"\s+",
                header=None,
                names=self.columns,
                na_values=self.fill_values,
                low_memory=False,
            )

            logger.info(f"  Loaded {len(df):,} records from {filepath.name}")

            # Convert DOY to datetime
            logger.info("  Converting DOY to datetime...")
            df["timestamp"] = df.apply(
                lambda row: self.doy_to_datetime(row["Year"], row["DOY"], row["Hour"]),
                axis=1,
            )

            # Remove invalid timestamps
            invalid_ts = df["timestamp"].isna().sum()
            if invalid_ts > 0:
                logger.warning(f"  Removed {invalid_ts} rows with invalid timestamps")
                df = df.dropna(subset=["timestamp"])

            # Keep only required columns
            cols_to_keep = [col for col in self.keep_cols if col in df.columns]
            df = df[cols_to_keep]

            logger.info(f"  Retained {len(cols_to_keep)} columns")

            return df

        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            raise

    def load_all_years(self, save=True):
        """
        Load all OMNI2 files and combine into single DataFrame.

        Args:
            save: Whether to save the combined data (default True)

        Returns:
            Combined DataFrame with all years
        """
        logger.info("=" * 70)
        logger.info("STARTING OMNI DATA LOADING PROCESS")
        logger.info("=" * 70)

        # Find all OMNI files
        omni_files = self.find_omni_files()

        # Load each file
        all_data = []

        for filepath in tqdm(omni_files, desc="Loading OMNI files"):
            try:
                df = self.load_single_file(filepath)
                all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to load {filepath.name}: {e}")
                continue

        if not all_data:
            raise ValueError("No data was loaded from any file!")

        # Combine all years
        logger.info("\nCombining all years...")
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df):,} records")

        # Remove duplicates
        logger.info("Removing duplicate timestamps...")
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="first")
        duplicates_removed = initial_count - len(combined_df)
        logger.info(f"Removed {duplicates_removed:,} duplicate records")

        # Sort by timestamp
        logger.info("Sorting by timestamp...")
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        # Data summary
        logger.info("\n" + "=" * 70)
        logger.info("OMNI DATA LOADING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total records loaded: {len(combined_df):,}")
        logger.info(
            f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}"
        )
        logger.info(f"Number of columns: {len(combined_df.columns)}")
        logger.info(f"Columns: {', '.join(combined_df.columns)}")

        # Missing data summary
        missing_pct = (combined_df.isna().sum() / len(combined_df) * 100).round(2)
        logger.info("\nMissing Data Summary:")
        for col in combined_df.columns:
            if col != "timestamp":
                logger.info(f"  {col:20s}: {missing_pct[col]:6.2f}%")

        # Save if requested
        if save:
            self.save_data(combined_df)

        logger.info("=" * 70)
        logger.info("OMNI DATA LOADING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return combined_df

    def save_data(self, df):
        """
        Save the combined dataset as Parquet and CSV.

        Args:
            df: DataFrame to save
        """
        logger.info("\nSaving processed data...")

        # Save as Parquet (efficient binary format)
        logger.info(f"Saving Parquet: {OMNI_PARQUET}")
        df.to_parquet(OMNI_PARQUET, index=False, compression="snappy")
        parquet_size = OMNI_PARQUET.stat().st_size / (1024**2)
        logger.info(f"  Parquet file size: {parquet_size:.2f} MB")

        # Save as CSV (human-readable)
        logger.info(f"Saving CSV: {OMNI_CSV}")
        df.to_csv(OMNI_CSV, index=False)
        csv_size = OMNI_CSV.stat().st_size / (1024**2)
        logger.info(f"  CSV file size: {csv_size:.2f} MB")

        logger.info(f"  Compression ratio: {csv_size/parquet_size:.2f}x")
        logger.info("✅ Data saved successfully!")


def load_omni_data(reload=False):
    """
    Convenience function to load OMNI data.
    Loads from saved file if exists, otherwise loads raw files.

    Args:
        reload: Force reload from raw .dat files (default False)

    Returns:
        DataFrame with OMNI data
    """
    if OMNI_PARQUET.exists() and not reload:
        logger.info(f"Loading existing OMNI data from {OMNI_PARQUET}")
        df = pd.read_parquet(OMNI_PARQUET)
        logger.info(f"Loaded {len(df):,} records")
        return df
    else:
        logger.info("No saved OMNI data found or reload requested")
        loader = OMNIDataLoader()
        return loader.load_all_years(save=True)


if __name__ == "__main__":
    """
    Run data loading when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - OMNI DATA LOADER")
    print("=" * 70 + "\n")

    try:
        loader = OMNIDataLoader()
        df = loader.load_all_years(save=True)

        print("\n✅ SUCCESS!")
        print(f"Loaded and saved {len(df):,} records")
        print(f"Files saved to: {PROCESSED_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Data loading failed")
        sys.exit(1)
