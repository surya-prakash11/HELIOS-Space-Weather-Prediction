"""
═══════════════════════════════════════════════════════════════
HELIOS - Master Pipeline Runner
═══════════════════════════════════════════════════════════════
Executes complete data pipeline from raw data to trained models

Pipeline Steps:
1. Load OMNI data
2. Clean data
3. Merge with flare data
4. Feature engineering
5. EDA
6. Train classifiers
7. K-Means clustering
8. Association rule mining

Usage:
    python run/pipeline.py           # Run all steps
    python run/pipeline.py --skip-to 4  # Start from step 4
═══════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import time
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    setup_logging,
    print_banner,
    PROCESSED_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    WAREHOUSE_PARQUET,
)

logger = setup_logging("pipeline")


class HELIOSPipeline:
    """
    Master pipeline orchestrator for HELIOS project.

    Executes all steps in order:
    1. Data loading
    2. Data cleaning
    3. Data merging
    4. Feature engineering
    5. EDA
    6. Model training
    7. K-Means clustering
    8. Association rule mining

    Usage:
        pipeline = HELIOSPipeline()
        pipeline.run_all()
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.steps = {
            1: ("Load OMNI Data", self.step_1_load_data),
            2: ("Clean Data", self.step_2_clean_data),
            3: ("Merge Datasets", self.step_3_merge_data),
            4: ("Feature Engineering", self.step_4_engineer_features),
            5: ("Exploratory Data Analysis", self.step_5_eda),
            6: ("Train Classifiers", self.step_6_train_models),
            7: ("K-Means Clustering", self.step_7_clustering),
            8: ("Association Rule Mining", self.step_8_association_rules),
        }

        self.start_time = None

    def step_1_load_data(self):
        """Step 1: Load OMNI data."""
        from src.data_loader import OMNIDataLoader

        loader = OMNIDataLoader()
        df = loader.load_all_years(save=True)
        return df

    def step_2_clean_data(self):
        """Step 2: Clean data."""
        from src.data_cleaner import clean_omni_data

        df_clean = clean_omni_data()
        return df_clean

    def step_3_merge_data(self):
        """Step 3: Merge OMNI and flare data."""
        from src.merger import merge_and_create_warehouse

        warehouse_df = merge_and_create_warehouse()
        return warehouse_df

    def step_4_engineer_features(self):
        """Step 4: Feature engineering."""
        from src.feature_engineer import engineer_features
        import pandas as pd

        # Load warehouse data
        df = pd.read_parquet(WAREHOUSE_PARQUET)

        # Engineer features
        df_features = engineer_features(df)

        # Save back to warehouse
        df_features.to_parquet(WAREHOUSE_PARQUET, index=False)

        return df_features

    def step_5_eda(self):
        """Step 5: Exploratory Data Analysis."""
        from src.eda import run_eda

        run_eda()

    def step_6_train_models(self):
        """Step 6: Train classification models."""
        from src.models.classifier import FlareClassifier

        classifier = FlareClassifier()
        results = classifier.train_all_models()
        return results

    def step_7_clustering(self):
        """Step 7: K-Means clustering."""
        from src.models.clustering import run_clustering

        clusterer = run_clustering()
        return clusterer

    def step_8_association_rules(self):
        """Step 8: Association rule mining."""
        from src.models.association_rules import run_association_mining

        miner, rules = run_association_mining()
        return miner, rules

    def run_step(self, step_num):
        """
        Run a single pipeline step.

        Args:
            step_num: Step number (1-8)
        """
        if step_num not in self.steps:
            raise ValueError(f"Invalid step number: {step_num}")

        step_name, step_func = self.steps[step_num]

        logger.info("\n" + "=" * 70)
        logger.info(f"STEP {step_num}/8: {step_name.upper()}")
        logger.info("=" * 70)

        step_start = time.time()

        try:
            result = step_func()
            step_elapsed = time.time() - step_start

            logger.info(f"\n✅ Step {step_num} completed in {step_elapsed:.2f} seconds")

            return result

        except Exception as e:
            logger.error(f"\n❌ Step {step_num} failed: {e}")
            logger.exception("Step failed with exception")
            raise

    def run_all(self, start_from=1):
        """
        Run complete pipeline.

        Args:
            start_from: Step number to start from (default 1)
        """
        print_banner()

        logger.info("\n" + "╔" + "=" * 68 + "╗")
        logger.info(
            "║" + " " * 15 + "HELIOS MASTER PIPELINE EXECUTION" + " " * 21 + "║"
        )
        logger.info("╚" + "=" * 68 + "╝")

        if start_from > 1:
            logger.info(
                f"\n⚠️  Starting from step {start_from} (skipping steps 1-{start_from-1})"
            )

        self.start_time = time.time()

        results = {}

        for step_num in range(start_from, 9):  # Changed from 7 to 9
            try:
                result = self.run_step(step_num)
                results[step_num] = result

            except Exception as e:
                logger.error(f"\n❌ Pipeline failed at step {step_num}")
                logger.error("Please fix the error and restart the pipeline")
                sys.exit(1)

        total_elapsed = time.time() - self.start_time

        # Final summary
        logger.info("\n" + "╔" + "=" * 68 + "╗")
        logger.info(
            "║" + " " * 20 + "PIPELINE COMPLETED SUCCESSFULLY!" + " " * 17 + "║"
        )
        logger.info("╚" + "=" * 68 + "╝")

        logger.info(
            f"\n⏱️  Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)"
        )

        logger.info("\n📁 Output Locations:")
        logger.info(f"  Processed Data:    {PROCESSED_DIR}")
        logger.info(f"  Warehouse Data:    {WAREHOUSE_PARQUET}")
        logger.info(f"  Trained Models:    {MODELS_DIR}")
        logger.info(f"  Visualizations:    {VISUALIZATIONS_DIR}")

        logger.info("\n🚀 Next Steps:")
        logger.info("  1. Review visualizations in the 'visualizations' folder")
        logger.info("  2. Check model performance in 'reports/model_comparison.csv'")
        logger.info("  3. Review clustering results in 'reports/cluster_profiles.csv'")
        logger.info("  4. Check association rules in 'reports/association_rules.csv'")
        logger.info("  5. Launch dashboard: streamlit run dashboard/app.py")

        return results


def main():
    """Main entry point for pipeline."""
    parser = argparse.ArgumentParser(
        description="HELIOS Master Pipeline - Complete data processing and model training"
    )
    parser.add_argument(
        "--skip-to",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Start pipeline from specific step (1-8)",
    )

    args = parser.parse_args()

    try:
        pipeline = HELIOSPipeline()
        pipeline.run_all(start_from=args.skip_to)

        print("\n" + "=" * 70)
        print("✅ HELIOS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📊 Review 2 Deliverables Complete:")
        print("  ✅ ETL Pipeline")
        print("  ✅ Star Schema Data Warehouse")
        print("  ✅ OLAP Operations")
        print("  ✅ Comprehensive EDA (6 plot types)")
        print("  ✅ Random Forest Classifier")
        print("  ✅ XGBoost Classifier")
        print("  ✅ K-Means Clustering")
        print("  ✅ Association Rule Mining")
        print("\n" + "=" * 70)

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        logger.exception("Pipeline execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
