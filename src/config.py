"""Paths, constants, and shared configuration."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Raw files
BONDDATA_CSV = DATA_RAW / "bonddata.csv"
MIKEY_CSV = DATA_RAW / "mikey.csv"
FH_2026_CSV = DATA_RAW / "FinancialHighlights_2026.csv"
FH_5Y_DIR = DATA_RAW / "fundamentals_5y"

# Interim parquet outputs
BONDS_PARQUET = DATA_INTERIM / "bonds.parquet"
MIKEY_PARQUET = DATA_INTERIM / "mikey_crosswalk.parquet"
FUNDAMENTALS_PARQUET = DATA_INTERIM / "fundamentals_panel.parquet"

# Processed feature matrices
FEATURES_SPREADS = DATA_PROCESSED / "feature_matrix_spreads.parquet"
FEATURES_DOWNGRADES = DATA_PROCESSED / "feature_matrix_downgrades.parquet"

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Bond data export date — used as as_of_date when not embedded in file
BOND_AS_OF_DATE = "2026-05-27"

# Spread model target column (z_spread_bid; fall back to g_spread_bid)
SPREAD_TARGET = "z_spread_bid"
SPREAD_FALLBACK = "g_spread_bid"

# Spread sanity bounds (bps) — outside this range = data error
# Z-spread can be slightly negative for short-dated IG bonds near par
SPREAD_MIN_BPS = -100.0
SPREAD_MAX_BPS = 2000.0

# Keep OAS aliases for backward compat in tests
OAS_MIN_BPS = SPREAD_MIN_BPS
OAS_MAX_BPS = SPREAD_MAX_BPS

# Maximum quarters a fundamental can be stale before we drop it
MAX_STALENESS_QUARTERS = 2

# US stock exchanges — used to filter mikey crosswalk to US issuers only
US_EXCHANGES = {"NYSE", "NASDAQGS", "NASDAQCM", "NASDAQGM", "NYSEAM", "NASDAQSC"}

# ---------------------------------------------------------------------------
# S&P rating numeric scale — lower number = higher quality
# ---------------------------------------------------------------------------
SP_RATING_SCALE = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
}

# Ratings at or below this numeric value are Investment Grade
IG_CUTOFF_NUMERIC = SP_RATING_SCALE["BBB-"]  # 10

# Ratings treated as NaN for numeric conversion
RATING_NA_STRINGS = {"NR", "WD", "SD", "", "N/A", "nan", "NaN"}

# ---------------------------------------------------------------------------
# FRED series for macro overlay
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "treasury_10y": "DGS10",
    "vix": "VIXCLS",
    "yield_curve": "T10Y2Y",
}
