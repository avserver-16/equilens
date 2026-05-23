import pandas as pd
import numpy as np
import json
import os
from modules.factors import run_factor_engine

def load_scorer_config() -> dict:
    with open("config.json") as f:
        return json.load(f)


def assign_conviction_tier(df: pd.DataFrame,
                            cfg: dict) -> pd.Series:
    sc = cfg["scorer"]
    n  = len(df)

    # Percentile-based cutoffs — with a minimum floor
    # so tiers work even with small universes
    tier_a_cut = max(10, int(n * sc["tier_a_percentile"]))
    tier_b_cut = max(30, int(n * sc["tier_b_percentile"]))
    tier_c_cut = max(60, int(n * sc["tier_c_percentile"]))
    min_pos    = sc["min_pillars_positive"]

    def pillars_positive(row):
        return sum([
            1 if (pd.notna(row["score_valuation"]) and row["score_valuation"] > 0) else 0,
            1 if (pd.notna(row["score_quality"])   and row["score_quality"]   > 0) else 0,
            1 if (pd.notna(row["score_momentum"])  and row["score_momentum"]  > 0) else 0,
        ])

    df = df.copy()
    df["pillars_positive"] = df.apply(pillars_positive, axis=1)

    tiers = []
    for _, row in df.iterrows():
        rank = row["rank"]
        pp   = row["pillars_positive"]
        if rank <= tier_a_cut and pp >= min_pos:
            tiers.append("A")
        elif rank <= tier_b_cut and pp >= min_pos:
            tiers.append("B")
        elif rank <= tier_c_cut:
            tiers.append("C")
        else:
            tiers.append(None)

    return pd.Series(tiers, index=df.index)
def build_watchlist(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construct the final watchlist by walking down the ranked list
    and applying sector concentration rules.

    Rules (from config):
    - Max `max_per_sector` stocks from any single GICS sector
    - Keep adding until `watchlist_size` stocks selected
    - At least `min_sectors_covered` distinct sectors in final list

    Returns a DataFrame of the selected stocks only.
    """
    sc              = cfg["scorer"]
    target_size     = sc["watchlist_size"]
    max_per_sector  = sc["max_per_sector"]
    min_sectors     = sc["min_sectors_covered"]

    sector_counts = {}
    selected_idx  = []

    # Walk ranked list from best to worst
    for idx, row in df.iterrows():
        sector = row.get("sector", "Unknown")
        count  = sector_counts.get(sector, 0)

        if count < max_per_sector:
            selected_idx.append(idx)
            sector_counts[sector] = count + 1

        if len(selected_idx) >= target_size:
            break

    watchlist = df.loc[selected_idx].copy()

    # Sector coverage check — warn if constraint not met
    sectors_covered = watchlist["sector"].nunique()
    if sectors_covered < min_sectors:
        print(f"  [WARN] Only {sectors_covered} sectors in watchlist "
              f"(target >= {min_sectors}). "
              f"Consider reducing max_per_sector.")
    else:
        print(f"  Sector coverage: {sectors_covered} sectors ✓")

    return watchlist.reset_index(drop=True)
def compute_confidence_score(row: pd.Series) -> int:
    """
    Convert composite z-score to a 0–100 confidence integer.
    Used in the investment memo (Module 7).

    Maps the composite_score distribution to a 0–100 range
    using min-max normalisation anchored at ±3 std.
    Score of 0 = worst ranked, 100 = best ranked.
    """
    raw = row["composite_score"]
    if pd.isna(raw):
        return 0
    # Clamp to [-3, +3] then scale to [0, 100]
    clamped = max(-3.0, min(3.0, raw))
    return int(round((clamped + 3.0) / 6.0 * 100))


def run_scorer(save: bool = True) -> pd.DataFrame:
    """
    Full Module 4 pipeline:
    1. Run the Factor Engine (Module 3) to get ranked scores
    2. Assign conviction tiers
    3. Build sector-diversified 20-stock watchlist
    4. Add confidence scores (0–100 int) for memo generator
    5. Save watchlist CSV and print summary

    Returns the watchlist DataFrame.
    """
    cfg = load_scorer_config()

    # ── Step 1: get ranked scores from Factor Engine ─────────────────────────
    print("=" * 55)
    print("MODULE 4 — COMPOSITE SCORER")
    print("=" * 55)
    df = run_factor_engine(
        min_market_cap_B=cfg["universe"]["min_market_cap_B"]
    )

    # ── Step 2: conviction tiers ─────────────────────────────────────────────
    print("\nAssigning conviction tiers...")
    df["tier"] = assign_conviction_tier(df, cfg)

    tier_counts = df["tier"].value_counts(dropna=False)
    print(f"  Tier A: {tier_counts.get('A', 0)} stocks")
    print(f"  Tier B: {tier_counts.get('B', 0)} stocks")
    print(f"  Tier C: {tier_counts.get('C', 0)} stocks")

    # ── Step 3: build diversified watchlist ──────────────────────────────────
    print("\nBuilding sector-diversified watchlist...")
    watchlist = build_watchlist(df, cfg)

    # ── Step 4: confidence scores ────────────────────────────────────────────
    watchlist["confidence"] = watchlist.apply(
        compute_confidence_score, axis=1
    )

    # ── Step 5: clean display columns ────────────────────────────────────────
    display_cols = [
        "rank", "ticker", "sector", "tier",
        "score_valuation", "score_quality", "score_momentum",
        "composite_score", "confidence",
        "pe_ratio", "ev_ebitda", "roic",
        "price_3m", "earnings_growth"
    ]
    # Only keep cols that exist (some may be missing if ingestion was partial)
    display_cols = [c for c in display_cols if c in watchlist.columns]

    print("\n" + "=" * 55)
    print("FINAL WATCHLIST — TOP 20")
    print("=" * 55)
    print(watchlist[display_cols].to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────────
    if save:
        os.makedirs("data", exist_ok=True)
        path = cfg["output"]["watchlist_csv"]
        watchlist.to_csv(path, index=False)
        print(f"\nWatchlist saved to {path}")

        # Also save the full scored universe for Module 6 (divergence)
        df.to_csv(cfg["output"]["scores_csv"], index=False)
        print(f"Full scores saved to {cfg['output']['scores_csv']}")

    return watchlist


# ── Utility: load cached watchlist (used by Module 7 — Memo Generator) ───────
def load_watchlist() -> pd.DataFrame:
    """
    Load the saved watchlist CSV.
    Module 7 calls this instead of re-running the full pipeline.
    """
    with open("config.json") as f:
        cfg = json.load(f)
    path = cfg["output"]["watchlist_csv"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Watchlist not found at {path}. Run scorer.py first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    watchlist = run_scorer()