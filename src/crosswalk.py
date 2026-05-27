"""
Build the issuer-name → MI Key crosswalk.

mikey.csv contains many non-US tickers (ASX, TSX, etc.) for the same
ticker symbol. We filter to US exchanges only before matching.
Issuers with no US-exchange match are silently excluded from the panel.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import DATA_INTERIM, MIKEY_CSV, MIKEY_PARQUET, US_EXCHANGES

_US_EXCHANGE_RE = re.compile(
    r"\((" + "|".join(re.escape(e) for e in US_EXCHANGES) + r"):"
)


def _normalize_name(name: str) -> str:
    """Strip exchange suffix and legal-entity suffix; lowercase."""
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name)
    name = re.sub(
        r",?\s*(inc\.?|corp\.?|corporation|co\.?|ltd\.?|llc|l\.p\.|lp|plc)\.?\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return name.strip().lower()


def parse_mikey(path: Path = MIKEY_CSV) -> pd.DataFrame:
    """
    Parse mikey.csv → crosswalk DataFrame (US exchanges only).

    File structure:
      rows 0-2: metadata / export timestamp
      row 3:    header (Ticker, Entity Name, Last Time, MI KEY)
      row 4+:   data
    """
    df = pd.read_csv(path, skiprows=[0, 1, 2], header=0, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[["Ticker", "Entity Name", "MI KEY"]].copy()
    df.columns = ["ticker", "entity_name", "mi_key"]

    df = df[df["ticker"].notna() & ~df["ticker"].str.strip().str.lower().eq("ticker")]
    df = df[df["entity_name"].notna()]
    df = df[df["mi_key"].notna()]

    df["mi_key"] = pd.to_numeric(df["mi_key"], errors="coerce").astype("Int64").astype(str)
    df = df[df["mi_key"] != "<NA>"]

    # US exchanges only
    df["is_us"] = df["entity_name"].apply(
        lambda n: bool(_US_EXCHANGE_RE.search(str(n)))
    )
    df_us = df[df["is_us"]].copy()
    df_us["entity_name_clean"] = df_us["entity_name"].apply(_normalize_name)
    df_us = df_us.drop_duplicates(subset=["mi_key"]).reset_index(drop=True)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df_us.to_parquet(MIKEY_PARQUET, index=False)
    print(f"mikey_crosswalk.parquet: {len(df_us)} US issuers")
    return df_us


def build_crosswalk(
    bonds_df: pd.DataFrame,
    mikey_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Match bonddata issuer names to MI Keys.

    Strategy:
      1. Exact match on normalized name
      2. Prefix / substring match
      Issuers with no match are excluded (logged as warnings).

    Returns DataFrame with columns: issuer_name, mi_key, match_type.
    """
    if mikey_df is None:
        mikey_df = parse_mikey()

    issuers = bonds_df["issuer_name"].dropna().unique()
    lookup: dict[str, tuple[str, str]] = {}

    for _, row in mikey_df.iterrows():
        norm = row["entity_name_clean"]
        lookup[norm] = (row["mi_key"], "exact")

    results = []
    unmatched = []

    for issuer in issuers:
        norm = _normalize_name(issuer)

        if norm in lookup:
            mi_key, mtype = lookup[norm]
            results.append({"issuer_name": issuer, "mi_key": mi_key, "match_type": mtype})
            continue

        matched = None
        for cand_norm, (cand_key, _) in lookup.items():
            if norm.startswith(cand_norm) or cand_norm.startswith(norm):
                matched = (cand_key, "prefix")
                break
        if matched:
            results.append({"issuer_name": issuer, "mi_key": matched[0], "match_type": matched[1]})
            continue

        unmatched.append(issuer)

    if unmatched:
        print(
            f"INFO: {len(unmatched)} issuers excluded (no US-exchange MI Key match):\n"
            + "\n".join(f"  {u!r}" for u in unmatched)
        )

    return pd.DataFrame(results)


def run_crosswalk(bonds_df: pd.DataFrame) -> pd.DataFrame:
    """Parse mikey, build crosswalk, return crosswalk DataFrame."""
    mikey_df = parse_mikey()
    cw = build_crosswalk(bonds_df, mikey_df)
    print(
        f"Crosswalk: {len(cw)} issuers mapped "
        f"({(cw['match_type']=='exact').sum()} exact, "
        f"{(cw['match_type']=='prefix').sum()} prefix)"
    )
    return cw
