"""
═══════════════════════════════════════════════════════════════
HELIOS - Heliophysics Event Learning & Intelligent Observation System
Configuration Module
═══════════════════════════════════════════════════════════════
Author: Surya Prakash Sankarakuthalam
Reg No: 24BCE7712
Course: CSE4005 - Data Warehousing and Data Mining
        CSE1006 - Foundations for Data Analytics
University: VIT University
═══════════════════════════════════════════════════════════════
"""

from pathlib import Path
import logging
import sys

# ═══════════════════════════════════════════════════════════════
# PROJECT METADATA
# ═══════════════════════════════════════════════════════════════

PROJECT_NAME = "HELIOS"
FULL_NAME = "Heliophysics Event Learning & Intelligent Observation System"
AUTHOR = "Surya Prakash Sankarakuthalam"
REG_NO = "24BCE7712"
UNIVERSITY = "VIT University"
VERSION = "2.0.0"

# ═══════════════════════════════════════════════════════════════
# DIRECTORY STRUCTURE
# ═══════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"
REPORTS_DIR = ROOT_DIR / "reports"

# Create all directories
for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════

# OMNI Data Files
OMNI_PARQUET = PROCESSED_DIR / "omni_full_2010_2025.parquet"
OMNI_CSV = PROCESSED_DIR / "omni_full_2010_2025.csv"

# Solar Flare Data
FLARE_RAW = RAW_DIR / "solar_flares.csv"

# Warehouse Files
WAREHOUSE_PARQUET = PROCESSED_DIR / "helios_warehouse.parquet"
WAREHOUSE_CSV = PROCESSED_DIR / "helios_warehouse.csv"

# Model Files
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Log File
LOG_FILE = ROOT_DIR / "helios.log"

# ═══════════════════════════════════════════════════════════════
# OMNI2 DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Complete OMNI2 column names (43 columns)
OMNI_COLUMNS = [
    "Year",
    "DOY",
    "Hour",
    "Minute",
    "ID_IMF",
    "ID_Plasma",
    "N_IMF_avg",
    "N_Plasma_avg",
    "Percent_interp",
    "Timeshift",
    "RMS_Timeshift",
    "RMS_phase",
    "Time_btwn_obs",
    "IMF_B_Avg",
    "Bx_GSE",
    "By_GSM",
    "Bz_GSM",
    "By_GSE",
    "Bz_GSE",
    "RMS_SD_B",
    "RMS_SD_BV",
    "SW_Speed",
    "Vx_GSE",
    "Vy_GSE",
    "Vz_GSE",
    "Proton_Density",
    "Temperature",
    "Flow_Pressure",
    "Electric_Field",
    "Plasma_Beta",
    "Alfven_Mach",
    "Kp_Index",
    "R_Sunspot",
    "Dst_Index",
    "AE_Index",
    "Proton_Flux_1MeV",
    "Proton_Flux_2MeV",
    "Proton_Flux_4MeV",
    "Proton_Flux_10MeV",
    "Proton_Flux_30MeV",
    "Proton_Flux_60MeV",
    "Flag",
]

# Fill values indicating missing data
OMNI_FILL_VALUES = [
    9999.99,
    99999.9,
    999.99,
    9999.9,
    999999.9,
    9999999.0,
    99.99,
    9999,
    99999,
    9999999,
]

# Columns to keep after loading
OMNI_KEEP_COLS = [
    "timestamp",
    "Year",
    "DOY",
    "Hour",
    "IMF_B_Avg",
    "Bx_GSE",
    "By_GSE",
    "Bz_GSE",
    "Bz_GSM",
    "SW_Speed",
    "Proton_Density",
    "Temperature",
    "Flow_Pressure",
    "Electric_Field",
    "Kp_Index",
    "R_Sunspot",
    "Dst_Index",
    "AE_Index",
    "Proton_Flux_10MeV",
]

# Year range for OMNI data
OMNI_START_YEAR = 2010
OMNI_END_YEAR = 2025

# ═══════════════════════════════════════════════════════════════
# SOLAR FLARE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Flare class encoding
FLARE_CLASS_ENCODING = {
    "A": 1,
    "B": 2,
    "C": 3,
    "M": 4,
    "X": 5,
    "No Flare": 0,
    None: 0,
    "": 0,
}

# Reverse mapping
FLARE_CLASS_DECODING = {
    v: k for k, v in FLARE_CLASS_ENCODING.items() if k not in [None, ""]
}
FLARE_CLASS_DECODING[0] = "No Flare"

# Flare class descriptions
FLARE_CLASS_DESCRIPTIONS = {
    0: "No Flare - Normal solar conditions",
    1: "A-Class - Background level, no significant effects",
    2: "B-Class - Minor, no significant impacts",
    3: "C-Class - Small, minor radio blackouts at poles",
    4: "M-Class - Medium, radio blackouts and radiation storms",
    5: "X-Class - Major, widespread radio blackouts and satellite damage",
}

# Flare severity levels
FLARE_SEVERITY_LEVELS = {
    0: "None",
    1: "Minimal",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Extreme",
}

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [3, 6, 24]

# Lag periods (in hours)
LAG_PERIODS = [1, 3, 6, 12, 24]

# Features for rolling statistics
ROLLING_FEATURES = [
    "SW_Speed",
    "Bz_GSE",
    "IMF_B_Avg",
    "Proton_Density",
    "solar_wind_energy",
    "bz_magnitude",
]

# Features for lag computation
LAG_FEATURES = [
    "SW_Speed",
    "Bz_GSE",
    "Proton_Density",
    "solar_wind_energy",
    "Kp_Index",
    "Dst_Index",
]

# ML feature columns (used for training)
ML_FEATURE_COLUMNS = [
    "IMF_B_Avg",
    "Bx_GSE",
    "By_GSE",
    "Bz_GSE",
    "SW_Speed",
    "Proton_Density",
    "Flow_Pressure",
    "Electric_Field",
    "Kp_Index",
    "Dst_Index",
    "solar_wind_energy",
    "bz_magnitude",
    "imf_total",
    "temp_proxy",
    "SW_Speed_3h_mean",
    "SW_Speed_6h_mean",
    "SW_Speed_24h_mean",
    "Bz_GSE_3h_mean",
    "Bz_GSE_6h_mean",
    "Bz_GSE_24h_mean",
    "SW_Speed_6h_std",
    "Bz_GSE_6h_std",
    "SW_Speed_delta_1h",
    "Bz_GSE_delta_1h",
    "hour",
    "month",
    "day_of_week",
    "day_of_year",
]

# Target column
TARGET_COLUMN = "flare_class_encoded"

# ═══════════════════════════════════════════════════════════════
# MACHINE LEARNING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 50,
    "min_samples_leaf": 25,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 1,
}

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
}

# K-Means parameters
KMEANS_PARAMS = {
    "n_clusters": 3,
    "n_init": 10,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

# Cluster names
CLUSTER_NAMES = {
    0: "Quiet Solar Wind",
    1: "Moderate Solar Wind",
    2: "Turbulent / Storm-Prone",
}

# Cluster colors
CLUSTER_COLORS = {0: "green", 1: "orange", 2: "red"}

# Association rules parameters
APRIORI_PARAMS = {
    "min_support": 0.01,
    "min_confidence": 0.3,
    "min_lift": 1.5,
    "max_len": 4,
}

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Plot settings
PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_FIGSIZE = (12, 8)

# Color palettes
COLOR_PALETTE_FLARES = {
    0: "#2ecc71",  # Green - No Flare
    1: "#3498db",  # Blue - A
    2: "#f39c12",  # Orange - B
    3: "#e67e22",  # Dark Orange - C
    4: "#e74c3c",  # Red - M
    5: "#c0392b",  # Dark Red - X
}

COLOR_PALETTE_SEQUENTIAL = "viridis"
COLOR_PALETTE_DIVERGING = "coolwarm"

# ═══════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════


def setup_logging(name: str = "HELIOS") -> logging.Logger:
    """
    Set up logging configuration for the project.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger


# ═══════════════════════════════════════════════════════════════
# DATA WAREHOUSE SCHEMA
# ═══════════════════════════════════════════════════════════════

# Star Schema Tables
STAR_SCHEMA = {
    "fact_table": "solar_events_fact",
    "dimensions": {
        "time_dim": [
            "time_id",
            "timestamp",
            "year",
            "month",
            "day",
            "hour",
            "day_of_week",
            "quarter",
            "is_weekend",
            "solar_cycle_phase",
            "season",
        ],
        "flare_class_dim": [
            "class_id",
            "class_name",
            "severity_level",
            "description",
            "typical_impact",
        ],
        "solar_wind_dim": [
            "sw_id",
            "speed_category",
            "density_category",
            "bz_polarity",
            "energy_level",
        ],
    },
}

# Binning categories for warehouse
SPEED_CATEGORIES = {"Slow": (0, 350), "Moderate": (350, 500), "Fast": (500, 1000)}

DENSITY_CATEGORIES = {"Low": (0, 8), "Medium": (8, 16), "High": (16, 100)}

BZ_POLARITY = {"Southward": (-100, -2), "Neutral": (-2, 2), "Northward": (2, 100)}

ENERGY_CATEGORIES = {"Low": (0, 1e-3), "Medium": (1e-3, 5e-3), "High": (5e-3, 1e10)}

# ═══════════════════════════════════════════════════════════════
# DASHBOARD CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DASHBOARD_CONFIG = {
    "title": "HELIOS - Solar Flare Intelligence System",
    "icon": "☀️",
    "layout": "wide",
    "theme": {
        "primaryColor": "#FF6B35",
        "backgroundColor": "#0f0f23",
        "secondaryBackgroundColor": "#1e1e3f",
        "textColor": "#eeeeee",
        "font": "sans-serif",
    },
    "port": 8501,
}

# Risk score thresholds
RISK_THRESHOLDS = {"calm": 30, "elevated": 60, "high": 80, "extreme": 100}

# ═══════════════════════════════════════════════════════════════
# VALIDATION CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Data quality thresholds
MAX_MISSING_PERCENT = 60  # Drop rows with >60% missing
OUTLIER_PERCENTILES = (1, 99)  # Winsorize at 1st and 99th percentile
MAX_FORWARD_FILL_HOURS = 6  # Maximum hours to forward fill

# Physical parameter valid ranges (for outlier detection)
VALID_RANGES = {
    "SW_Speed": (200, 1000),  # km/s
    "Proton_Density": (0.1, 100),  # n/cc
    "IMF_B_Avg": (0, 50),  # nT
    "Temperature": (1e4, 1e7),  # K
    "Kp_Index": (0, 9),
    "Dst_Index": (-500, 100),  # nT
    "AE_Index": (0, 2500),  # nT
}

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def print_banner():
    """Print the HELIOS project banner."""
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║                           HELIOS                              ║
║   Heliophysics Event Learning & Intelligent Observation       ║
║                          System v{VERSION}                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Author     : {AUTHOR:45s} ║
║  Reg No     : {REG_NO:45s} ║
║  University : {UNIVERSITY:45s} ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def get_omni_files():
    """
    Get list of OMNI data files in raw directory.

    Returns:
        List of Path objects for OMNI .dat files
    """
    omni_files = sorted(RAW_DIR.glob("omni2_*.dat"))
    return omni_files


# ═══════════════════════════════════════════════════════════════
# MODULE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_banner()
    print(f"\n📁 Root Directory: {ROOT_DIR}")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"📁 Visualizations Directory: {VISUALIZATIONS_DIR}")
    print(f"📁 Reports Directory: {REPORTS_DIR}")
    print(f"\n✅ Configuration loaded successfully!")
