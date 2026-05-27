"""
Tests for src/crosswalk.py.

Run: pytest tests/test_crosswalk.py -v
"""
import pytest
import pandas as pd

from src.crosswalk import _normalize_name, build_crosswalk, parse_mikey
from src.parse_capital_iq import parse_bonddata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bonds_df():
    return parse_bonddata()


@pytest.fixture(scope="module")
def mikey_df():
    return parse_mikey()


@pytest.fixture(scope="module")
def crosswalk_df(bonds_df, mikey_df):
    return build_crosswalk(bonds_df, mikey_df)


# ---------------------------------------------------------------------------
# mikey.csv tests
# ---------------------------------------------------------------------------

def test_mikey_us_only(mikey_df):
    """All entries in parsed mikey must be US-listed."""
    from src.config import US_EXCHANGES
    import re
    us_re = re.compile(r"\((" + "|".join(re.escape(e) for e in US_EXCHANGES) + r"):")
    non_us = mikey_df[~mikey_df["entity_name"].apply(lambda n: bool(us_re.search(str(n))))]
    assert len(non_us) == 0, f"Found {len(non_us)} non-US entries in mikey crosswalk"


def test_mikey_mi_key_numeric(mikey_df):
    numeric = pd.to_numeric(mikey_df["mi_key"], errors="coerce")
    assert numeric.notna().all(), "Some MI Keys are non-numeric"


def test_mikey_no_duplicates(mikey_df):
    dup = mikey_df.duplicated(subset=["mi_key"]).sum()
    assert dup == 0, f"{dup} duplicate MI Keys in crosswalk"


# ---------------------------------------------------------------------------
# crosswalk matching tests
# ---------------------------------------------------------------------------

def test_crosswalk_columns(crosswalk_df):
    assert set(crosswalk_df.columns) >= {"issuer_name", "mi_key", "match_type"}


def test_crosswalk_at_least_68_issuers(crosswalk_df):
    """At least 68 of 72 bond issuers must be matched (4 excluded by design)."""
    assert len(crosswalk_df) >= 68, \
        f"Only {len(crosswalk_df)} issuers matched; expected ≥ 68"


def test_crosswalk_no_manual_entries(crosswalk_df):
    """No manual overrides — any unmatched issuer is simply excluded."""
    manual = crosswalk_df[crosswalk_df["match_type"] == "manual"]
    assert len(manual) == 0, f"Unexpected manual entries: {manual}"


def test_crosswalk_all_matched_have_mi_key(crosswalk_df):
    missing = crosswalk_df[crosswalk_df["mi_key"].isna()]
    assert len(missing) == 0, f"Matched rows with null MI Key: {missing}"


# ---------------------------------------------------------------------------
# Name normalizer unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("3M Company (NYSE:MMM)", "3m company"),
    ("Amgen Inc. (NASDAQGS:AMGN)", "amgen"),
    ("The Boeing Company (NYSE:BA)", "the boeing company"),
    ("Altria Group, Inc.", "altria group"),
    ("APA Corporation (NASDAQGS:APA)", "apa"),
])
def test_normalize_name(raw, expected):
    assert _normalize_name(raw) == expected
