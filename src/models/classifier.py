"""
═══════════════════════════════════════════════════════════════
HELIOS - Flare Classification Models
═══════════════════════════════════════════════════════════════
Trains and evaluates multiple classifiers:
1. Random Forest Classifier
2. XGBoost Classifier

Handles class imbalance using:
- Class weights
- SMOTE oversampling (optional)
- Stratified sampling

Evaluation metrics:
- Confusion matrix
- Classification report
- Feature importance
- ROC curves (for binary)
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import (
    WAREHOUSE_PARQUET,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    SCALER_PATH,
    ML_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    RF_PARAMS,
    XGB_PARAMS,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    FLARE_CLASS_DECODING,
    COLOR_PALETTE_FLARES,
    PLOT_DPI,
    setup_logging,
)

logger = setup_logging("classifier")


class FlareClassifier:
    """
    Solar flare classification system.

    Features:
    - Multiple model support (RF, XGBoost)
    - Automatic feature selection
    - Class imbalance handling
    - Comprehensive evaluation
    - Model persistence

    Usage:
        classifier = FlareClassifier()
        classifier.train_random_forest(X_train, y_train)
        predictions = classifier.predict(X_test)
    """

    def __init__(self):
        """Initialize the classifier."""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = FLARE_CLASS_DECODING

        logger.info("Flare Classifier initialized")

    def load_data(self):
        """
        Load and prepare warehouse data for ML.

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Loading warehouse data...")

        if not WAREHOUSE_PARQUET.exists():
            raise FileNotFoundError(
                f"Warehouse data not found: {WAREHOUSE_PARQUET}\n"
                "Please run the ETL pipeline first!"
            )

        df = pd.read_parquet(WAREHOUSE_PARQUET)
        logger.info(f"Loaded {len(df):,} records")

        # Select features
        available_features = [f for f in ML_FEATURE_COLUMNS if f in df.columns]

        if len(available_features) == 0:
            raise ValueError("No ML features found in warehouse data!")

        logger.info(f"Using {len(available_features)} features for training")

        # Check target column
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")

        # Prepare X and y
        X = df[available_features].copy()
        y = df[TARGET_COLUMN].copy()

        # Handle missing values in features
        logger.info("Handling missing values in features...")
        initial_missing = X.isna().sum().sum()
        X = X.fillna(X.median())
        logger.info(f"  Filled {initial_missing:,} missing values with median")

        # Store feature names
        self.feature_names = available_features

        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution:\n{y.value_counts().sort_index()}")

        return X, y, available_features

    def prepare_train_test_split(self, X, y, use_smote=False):
        """
        Split data into train and test sets.

        Args:
            X: Features
            y: Target
            use_smote: Whether to apply SMOTE oversampling

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set:  {len(X_test):,} samples")

        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

        # Apply SMOTE if requested
        if use_smote and len(np.unique(y_train)) > 1:
            logger.info("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {len(X_train):,} samples")
            logger.info(
                f"New class distribution:\n{pd.Series(y_train).value_counts().sort_index()}"
            )

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING RANDOM FOREST CLASSIFIER")
        logger.info("=" * 70)

        logger.info(f"Parameters: {RF_PARAMS}")

        model = RandomForestClassifier(**RF_PARAMS)

        logger.info("Training model...")
        model.fit(X_train, y_train)

        self.models["random_forest"] = model

        logger.info("✅ Random Forest training completed")

        # Feature importance
        self.plot_feature_importance(model, "random_forest")

        return model

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING XGBOOST CLASSIFIER")
        logger.info("=" * 70)

        logger.info(f"Parameters: {XGB_PARAMS}")

        model = XGBClassifier(**XGB_PARAMS)

        logger.info("Training model with early stopping...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=30,
            verbose=False,
        )

        self.models["xgboost"] = model

        logger.info("✅ XGBoost training completed")

        # Feature importance
        self.plot_feature_importance(model, "xgboost")

        return model

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate trained model.

        Args:
            model_name: Name of model to evaluate
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found!")

        model = self.models[model_name]

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info(f"\n{model_name.upper()} PERFORMANCE:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")

        # Classification report
        logger.info(f"\nClassification Report:")
        report = classification_report(
            y_test,
            y_pred,
            target_names=[self.class_names[i] for i in sorted(np.unique(y_test))],
            zero_division=0,
        )
        print(report)

        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, model_name)

        metrics = {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Raw counts
        ax1 = axes[0]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=[self.class_names[i] for i in sorted(np.unique(y_true))],
            yticklabels=[self.class_names[i] for i in sorted(np.unique(y_true))],
            cbar_kws={"label": "Count"},
        )
        ax1.set_title(
            f"{model_name.title()} - Confusion Matrix (Counts)",
            fontweight="bold",
            fontsize=12,
        )
        ax1.set_xlabel("Predicted Class", fontweight="bold")
        ax1.set_ylabel("True Class", fontweight="bold")

        # Plot 2: Normalized
        ax2 = axes[1]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Greens",
            ax=ax2,
            xticklabels=[self.class_names[i] for i in sorted(np.unique(y_true))],
            yticklabels=[self.class_names[i] for i in sorted(np.unique(y_true))],
            cbar_kws={"label": "Percentage"},
        )
        ax2.set_title(
            f"{model_name.title()} - Confusion Matrix (Normalized)",
            fontweight="bold",
            fontsize=12,
        )
        ax2.set_xlabel("Predicted Class", fontweight="bold")
        ax2.set_ylabel("True Class", fontweight="bold")

        plt.tight_layout()

        cm_path = VISUALIZATIONS_DIR / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Confusion matrix saved to: {cm_path}")
        plt.close()

    def plot_feature_importance(self, model, model_name, top_n=20):
        """
        Plot feature importance.

        Args:
            model: Trained model
            model_name: Name of model
            top_n: Number of top features to show
        """
        logger.info(f"Plotting feature importance for {model_name}...")

        # Get feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return

        # Create DataFrame
        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax.barh(
            range(len(importance_df)), importance_df["importance"].values, color=colors
        )

        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["feature"].values)
        ax.set_xlabel("Importance", fontweight="bold")
        ax.set_title(
            f"{model_name.title()} - Top {top_n} Feature Importances",
            fontweight="bold",
            fontsize=14,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df["importance"].values)):
            ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)

        plt.tight_layout()

        importance_path = VISUALIZATIONS_DIR / f"feature_importance_{model_name}.png"
        plt.savefig(importance_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"✅ Feature importance plot saved to: {importance_path}")
        plt.close()

    def save_models(self):
        """Save trained models and scaler."""
        logger.info("\nSaving models...")

        if "random_forest" in self.models:
            joblib.dump(self.models["random_forest"], RF_MODEL_PATH)
            logger.info(f"  ✅ Random Forest saved to: {RF_MODEL_PATH}")

        if "xgboost" in self.models:
            joblib.dump(self.models["xgboost"], XGB_MODEL_PATH)
            logger.info(f"  ✅ XGBoost saved to: {XGB_MODEL_PATH}")

        joblib.dump(self.scaler, SCALER_PATH)
        logger.info(f"  ✅ Scaler saved to: {SCALER_PATH}")

    def train_all_models(self):
        """
        Complete training pipeline for all models.

        Returns:
            Dict with evaluation results
        """
        logger.info("=" * 70)
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("=" * 70)

        # Load data
        X, y, feature_names = self.load_data()

        # Split data
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(
            X, y, use_smote=False
        )

        # Train Random Forest
        self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model("random_forest", X_test, y_test)

        # Train XGBoost
        self.train_xgboost(X_train, y_train, X_test, y_test)
        xgb_metrics = self.evaluate_model("xgboost", X_test, y_test)

        # Save models
        self.save_models()

        # Create comparison report
        comparison_df = pd.DataFrame([rf_metrics, xgb_metrics])
        comparison_path = REPORTS_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\n✅ Model comparison saved to: {comparison_path}")

        logger.info("\n" + "=" * 70)
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        print("\nMODEL COMPARISON:")
        print(comparison_df.to_string(index=False))

        return {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "comparison": comparison_df,
        }


if __name__ == "__main__":
    """
    Run classifier training when executed directly.
    """
    print("\n" + "=" * 70)
    print("HELIOS - FLARE CLASSIFICATION MODELS")
    print("=" * 70 + "\n")

    try:
        classifier = FlareClassifier()
        results = classifier.train_all_models()

        print("\n✅ SUCCESS!")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        print(f"Reports saved to: {REPORTS_DIR}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Classifier training failed")
        sys.exit(1)
