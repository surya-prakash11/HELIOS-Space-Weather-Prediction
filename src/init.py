"""
HELIOS Source Package
Heliophysics Event Learning & Intelligent Observation System
"""

__version__ = "2.0.0"
__author__ = "Surya Prakash Sankarakuthalam"

from pathlib import Path

# Package root
SRC_DIR = Path(__file__).parent

__all__ = ["__version__", "__author__", "SRC_DIR"]
