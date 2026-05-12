"""
═══════════════════════════════════════════════════════════════
HELIOS - Association Rule Mining Module
═══════════════════════════════════════════════════════════════
Discovers patterns and rules in space weather data using Apriori

Rules Format: IF {condition} THEN {outcome}
Example: IF WindSpeed=Fast AND BZ=Southward THEN Flare=X-Class

Features:
- Discretizes continuous variables into categories
- Applies Apriori algorithm
- Filters rules for flare consequents
- Visualizes rule strength (support, confidence, lift)
- Generates actionable insights

Usage:
    miner = AssociationRuleMiner()
    rules = miner.mine_rules()
    miner.visualize_rules(rules)
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import (
    WAREHOUSE_PARQUET,
    APRIORI_PARAMS,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    PLOT_DPI,
    setup_logging,
)

logger = setup_logging("association_rules")


class AssociationRuleMiner:
    """
    Association rule mining for solar wind and flare patterns.

    Features:
    - Automatic binning of continuous variables
    - Apriori frequent itemset mining
    - Rule generation with lift/confidence filtering
    - Interactive visualizations

    Usage:
        miner = AssociationRuleMiner()
        rules = miner.mine_rules()
        miner.visualize_rules(rules)
    """

    def __init__(self):
        """Initialize the association rule miner."""
        self.min_support = APRIORI_PARAMS["min_support"]
        self.min_confidence = APRIORI_PARAMS["min_confidence"]
        self.min_lift = APRIORI_PARAMS["min_lift"]
        self.max_len = APRIORI_PARAMS["max_len"]

        logger.info("Association Rule Miner initialized")
        logger.info(
            f"Parameters: support={self.min_support}, confidence={self.min_confidence}, lift={self.min_lift}"
        )

    def load_data(self):
        """
        Load warehouse data.

        Returns:
            DataFrame
        """
        logger.info("Loading warehouse data...")

        if not WAREHOUSE_PARQUET.exists():
            raise FileNotFoundError(
                f"Warehouse data not found: {WAREHOUSE_PARQUET}\n"
                "Please run the ETL pipeline first!"
            )

        df = pd.read_parquet(WAREHOUSE_PARQUET)
        logger.info(f"Loaded {len(df):,} records")

        return df

    def discretize_features(self, df):
        """
        Discretize continuous features into categorical bins.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with discretized features
        """
        logger.info("\nDiscretizing continuous features into categories...")

        df_disc = pd.DataFrame()

        # Solar Wind Speed
        if "SW_Speed" in df.columns:
            df_disc["WindSpeed"] = pd.cut(
                df["SW_Speed"],
                bins=[0, 350, 500, 1000],
                labels=["Slow", "Moderate", "Fast"],
                include_lowest=True,
            )
            logger.info("  ✓ SW_Speed → WindSpeed (Slow/Moderate/Fast)")

        # Bz Component
        if "Bz_GSE" in df.columns:
            df_disc["BZ_State"] = pd.cut(
                df["Bz_GSE"],
                bins=[-100, -2, 2, 100],
                labels=["Southward", "Neutral", "Northward"],
                include_lowest=True,
            )
            logger.info("  ✓ Bz_GSE → BZ_State (Southward/Neutral/Northward)")

        # Proton Density
        if "Proton_Density" in df.columns:
            df_disc["Density"] = pd.cut(
                df["Proton_Density"],
                bins=[0, 8, 16, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )
            logger.info("  ✓ Proton_Density → Density (Low/Medium/High)")

        # IMF Magnitude
        if "IMF_B_Avg" in df.columns:
            df_disc["IMF_Strength"] = pd.cut(
                df["IMF_B_Avg"],
                bins=[0, 4, 8, 50],
                labels=["Weak", "Moderate", "Strong"],
                include_lowest=True,
            )
            logger.info("  ✓ IMF_B_Avg → IMF_Strength (Weak/Moderate/Strong)")

        # Kp Index
        if "Kp_Index" in df.columns:
            df_disc["Kp_Level"] = pd.cut(
                df["Kp_Index"],
                bins=[0, 3, 5, 9],
                labels=["Quiet", "Active", "Storm"],
                include_lowest=True,
            )
            logger.info("  ✓ Kp_Index → Kp_Level (Quiet/Active/Storm)")

        # Dst Index
        if "Dst_Index" in df.columns:
            df_disc["Dst_Level"] = pd.cut(
                df["Dst_Index"],
                bins=[-500, -50, -30, 100],
                labels=["Storm", "Disturbed", "Quiet"],
                include_lowest=True,
            )
            logger.info("  ✓ Dst_Index → Dst_Level (Storm/Disturbed/Quiet)")

        # Flare Class (target)
        if "flare_class" in df.columns:
            df_disc["Flare_Event"] = df["flare_class"].apply(
                lambda x: (
                    "No_Flare"
                    if x == "No Flare"
                    else (
                        "Flare_AB"
                        if x in ["A", "B"]
                        else (
                            "Flare_C"
                            if x == "C"
                            else "Flare_MX" if x in ["M", "X"] else "No_Flare"
                        )
                    )
                )
            )
            logger.info(
                "  ✓ flare_class → Flare_Event (No_Flare/Flare_AB/Flare_C/Flare_MX)"
            )

        # Remove rows with NaN in discretized features
        initial_rows = len(df_disc)
        df_disc = df_disc.dropna()
        removed_rows = initial_rows - len(df_disc)

        if removed_rows > 0:
            logger.info(
                f"\n  Removed {removed_rows:,} rows with missing values after discretization"
            )

        logger.info(
            f"\nDiscretized dataset: {len(df_disc):,} records × {len(df_disc.columns)} features"
        )

        return df_disc

    def create_transaction_format(self, df_disc):
        """
        Convert discretized DataFrame to transaction format for Apriori.

        Args:
            df_disc: Discretized DataFrame

        Returns:
            One-hot encoded DataFrame
        """
        logger.info("\nConverting to transaction format...")

        # Create transactions (each row is a set of items)
        transactions = []
        for idx, row in df_disc.iterrows():
            transaction = []
            for col in df_disc.columns:
                if pd.notna(row[col]):
                    transaction.append(f"{col}={row[col]}")
            transactions.append(transaction)

        # One-hot encode
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        logger.info(
            f"Transaction format: {len(df_encoded):,} records × {len(df_encoded.columns)} items"
        )

        return df_encoded

    def mine_rules(self, df=None):
        """
        Mine association rules using Apriori algorithm.

        Args:
            df: Input DataFrame (optional, will load if None)

        Returns:
            DataFrame with association rules
        """
        logger.info("=" * 70)
        logger.info("STARTING ASSOCIATION RULE MINING")
        logger.info("=" * 70)

        if df is None:
            df = self.load_data()

        # Discretize features
        df_disc = self.discretize_features(df)

        # Convert to transaction format
        df_encoded = self.create_transaction_format(df_disc)

        # Apply Apriori to find frequent itemsets
        logger.info(f"\nMining frequent itemsets (min_support={self.min_support})...")
        frequent_itemsets = apriori(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )

        if len(frequent_itemsets) == 0:
            logger.warning("No frequent itemsets found! Try lowering min_support.")
            return pd.DataFrame()

        logger.info(f"Found {len(frequent_itemsets):,} frequent itemsets")

        # Generate association rules
        logger.info(
            f"\nGenerating association rules (min_confidence={self.min_confidence})..."
        )
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=self.min_confidence
        )

        if len(rules) == 0:
            logger.warning("No rules found! Try lowering min_confidence.")
            return pd.DataFrame()

        logger.info(f"Generated {len(rules):,} rules")

        # Filter for flare-related rules (consequent contains Flare_Event)
        logger.info("\nFiltering for flare-related rules...")
        flare_rules = rules[
            rules["consequents"].apply(
                lambda x: any("Flare_Event=" in str(item) for item in x)
            )
        ]

        logger.info(f"Found {len(flare_rules):,} flare-related rules")

        # Filter by lift
        flare_rules = flare_rules[flare_rules["lift"] >= self.min_lift]
        logger.info(
            f"After lift filter (>={self.min_lift}): {len(flare_rules):,} rules"
        )

        # Sort by lift (descending)
        flare_rules = flare_rules.sort_values("lift", ascending=False)

        # Format antecedents and consequents as strings
        flare_rules["antecedents_str"] = flare_rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )
        flare_rules["consequents_str"] = flare_rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )

        # Display top rules
        logger.info("\n" + "=" * 70)
        logger.info("TOP 10 ASSOCIATION RULES (by Lift)")
        logger.info("=" * 70)

        display_cols = [
            "antecedents_str",
            "consequents_str",
            "support",
            "confidence",
            "lift",
        ]
        top_rules = flare_rules[display_cols].head(10)

        for idx, row in top_rules.iterrows():
            logger.info(f"\nRule {idx + 1}:")
            logger.info(f"  IF {row['antecedents_str']}")
            logger.info(f"  THEN {row['consequents_str']}")
            logger.info(
                f"  Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.2f}"
            )

        logger.info("\n" + "=" * 70)

        # Save rules to CSV
        rules_path = REPORTS_DIR / "association_rules.csv"
        flare_rules[display_cols].to_csv(rules_path, index=False)
        logger.info(f"\n✅ Association rules saved to: {rules_path}")

        return flare_rules

    def visualize_rules(self, rules):
        """
        Create visualizations for association rules.

        Args:
            rules: DataFrame with association rules
        """
        if len(rules) == 0:
            logger.warning("No rules to visualize!")
            return

        logger.info("\nCreating association rule visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Support vs Confidence (colored by Lift)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            rules["support"],
            rules["confidence"],
            c=rules["lift"],
            s=50,
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.set_xlabel("Support", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Confidence", fontweight="bold", fontsize=12)
        ax1.set_title(
            "Association Rules: Support vs Confidence", fontweight="bold", fontsize=14
        )
        ax1.grid(alpha=0.3)
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label("Lift", fontweight="bold")

        # Plot 2: Top 10 rules by Lift
        ax2 = axes[0, 1]
        top_10 = rules.nlargest(10, "lift")
        rule_labels = [f"Rule {i+1}" for i in range(len(top_10))]

        bars = ax2.barh(
            rule_labels, top_10["lift"].values, color="coral", edgecolor="black"
        )
        ax2.set_xlabel("Lift", fontweight="bold", fontsize=12)
        ax2.set_title("Top 10 Rules by Lift", fontweight="bold", fontsize=14)
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

        # Plot 3: Confidence distribution
        ax3 = axes[1, 0]
        ax3.hist(
            rules["confidence"], bins=20, color="skyblue", edgecolor="black", alpha=0.7
        )
        ax3.axvline(
            rules["confidence"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {rules['confidence'].mean():.3f}",
        )
        ax3.set_xlabel("Confidence", fontweight="bold", fontsize=12)
        ax3.set_ylabel("Frequency", fontweight="bold", fontsize=12)
        ax3.set_title("Rule Confidence Distribution", fontweight="bold", fontsize=14)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Plot 4: Support vs Lift
        ax4 = axes[1, 1]
        scatter2 = ax4.scatter(
            rules["support"],
            rules["lift"],
            c=rules["confidence"],
            s=50,
            cmap="plasma",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )
        ax4.set_xlabel("Support", fontweight="bold", fontsize=12)
        ax4.set_ylabel("Lift", fontweight="bold", fontsize=12)
        ax4.set_title(
            "Association Rules: Support vs Lift", fontweight="bold", fontsize=14
        )
        ax4.grid(alpha=0.3)
        ax4.axhline(
            y=1,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Lift=1 (no association)",
        )
        ax4.legend()
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label("Confidence", fontweight="bold")

        plt.suptitle(
            "HELIOS - Association Rule Mining Results", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        viz_path = VISUALIZATIONS_DIR / "association_rules_viz.png"
        plt.savefig(viz_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Visualization saved to: {viz_path}")
        plt.close()


def run_association_mining():
    """
    Convenience function to run complete association rule mining.
    """
    miner = AssociationRuleMiner()
    rules = miner.mine_rules()

    if len(rules) > 0:
        miner.visualize_rules(rules)

    return miner, rules


if __name__ == "__main__":
    """
    Run association rule mining when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - ASSOCIATION RULE MINING")
    print("=" * 70 + "\n")

    try:
        miner, rules = run_association_mining()

        print("\n✅ SUCCESS!")
        print(f"Found {len(rules):,} flare-related association rules")
        print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        print(f"Reports saved to: {REPORTS_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Association rule mining failed")
        sys.exit(1)
