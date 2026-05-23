import pandas as pd
import numpy as np
import json
from scipy import stats
from modules.ingest import load_fundamentals
from modules.universe import load_universe

# ── Factor definitions ───────────────────────────────────────────────────────
# Each entry: (column_name, direction)
# direction = +1 → higher is better (quality, momentum)
# direction = -1 → lower is better (valuation, debt)

VALUATION_FACTORS = [
    ("pe_ratio",    -1),
    ("ev_ebitda",   -1),
    ("pb_ratio",    -1),
]

QUALITY_FACTORS = [
    ("roic",           +1),
    ("roe",            +1),
    ("gross_margin",   +1),
    ("debt_to_equity", -1),
]

MOMENTUM_FACTORS = [
    ("price_1m",        +1),
    ("price_3m",        +1),
    ("price_6m",        +1),
    ("earnings_growth", +1),
]

# Load pillar weights from config.json
#Thresholds for winsorisation=0.05 on both the ends
def load_weights() -> dict:
    with open("config.json") as f:
        cfg = json.load(f)
    return cfg["factors"]
def winsorise(series: pd.Series,
              lower: float = 0.05,
              upper: float = 0.95) -> pd.Series:
    """
    Clip extreme values at the lower and upper percentiles.
    Operates on a single column of floats.
    NaN values are ignored and preserved.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore_series(series: pd.Series) -> pd.Series:
    """
    Compute z-scores for a series.
    Returns NaN where std == 0 (all values identical) or where input is NaN.
    Requires at least 3 non-NaN values to be meaningful.
    """
    clean = series.dropna()
    if len(clean) < 3 or clean.std() == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - clean.mean()) / clean.std()


def compute_factor_zscore(df: pd.DataFrame,
                           col: str,
                           direction: int,
                           group_col: str = "sector") -> pd.Series:
    """
    For a given factor column:
    1. Winsorise within each sector group
    2. Z-score within each sector group
    3. Multiply by direction (+1 or -1)

    Returns a pd.Series of z-scores aligned to df.index.
    """
    result = pd.Series(np.nan, index=df.index)

    for sector, group in df.groupby(group_col):
        idx = group.index
        col_data = group[col].copy()

        # Skip if fewer than 4 stocks in this sector have data
        if col_data.dropna().shape[0] < 4:
            continue

        winsorised = winsorise(col_data)
        zscored    = zscore_series(winsorised)
        result[idx] = zscored * direction

    return result
def compute_pillar_score(df: pd.DataFrame,
                          factors: list,
                          pillar_name: str) -> pd.Series:
    """
    Compute a composite score for one pillar (valuation / quality / momentum).

    Steps:
    1. Compute z-score for each factor in the pillar
    2. Average across factors (ignoring NaN — a stock missing 1 factor
       is not penalised, it just uses the average of the others)
    3. Re-z-score the pillar composite so all pillars are on the same scale

    Returns a pd.Series of pillar scores aligned to df.index.
    """
    factor_scores = pd.DataFrame(index=df.index)

    for col, direction in factors:
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not found — skipping")
            continue
        factor_scores[col] = compute_factor_zscore(df, col, direction)

    if factor_scores.empty:
        return pd.Series(np.nan, index=df.index)

    # Row-wise mean, ignoring NaN
    pillar_raw = factor_scores.mean(axis=1, skipna=True)

    # Re-z-score the pillar so all three pillars are comparable
    pillar_z = zscore_series(pillar_raw)

    print(f"  {pillar_name:12s} | "
          f"valid={pillar_z.notna().sum():3d} | "
          f"mean={pillar_z.mean():.3f} | "
          f"std={pillar_z.std():.3f}")

    return pillar_z
def run_factor_engine(min_market_cap_B: float = 2.0) -> pd.DataFrame:
    """
    Full pipeline:
    1. Load fundamentals from SQLite (Module 2)
    2. Merge with sector data from universe.csv (Module 1)
    3. Compute all three pillar scores
    4. Combine into composite score using config weights
    5. Rank stocks and return sorted DataFrame

    Returns DataFrame with original columns + pillar scores + composite score + rank.
    Called by Module 4 (Scorer) and Module 6 (Divergence Detector).
    """
    weights = load_weights()

    # ── Load and merge ───────────────────────────────────────────────────────
    fundamentals = load_fundamentals(min_market_cap_B)
    universe     = load_universe()[["ticker", "sector", "sub_industry"]]
    df = fundamentals.merge(universe, on="ticker", how="left")

    # Drop stocks with no sector mapping (shouldn't happen but defensive)
    df = df.dropna(subset=["sector"]).reset_index(drop=True)
    print(f"\nRunning Factor Engine on {len(df)} stocks across "
          f"{df['sector'].nunique()} sectors\n")

    # ── Pillar scores ────────────────────────────────────────────────────────
    print("Computing pillar scores:")
    df["score_valuation"] = compute_pillar_score(
        df, VALUATION_FACTORS, "VALUATION"
    )
    df["score_quality"] = compute_pillar_score(
        df, QUALITY_FACTORS, "QUALITY"
    )
    df["score_momentum"] = compute_pillar_score(
        df, MOMENTUM_FACTORS, "MOMENTUM"
    )

    # ── Composite score ──────────────────────────────────────────────────────
    w_val = weights["valuation_weight"]
    w_qua = weights["quality_weight"]
    w_mom = weights["momentum_weight"]

    df["composite_score"] = (
        df["score_valuation"].fillna(0) * w_val +
        df["score_quality"].fillna(0)   * w_qua +
        df["score_momentum"].fillna(0)  * w_mom
    )

    # ── Rank (1 = best) ──────────────────────────────────────────────────────
    df["rank"] = df["composite_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)

    df = df.sort_values("rank").reset_index(drop=True)

    print(f"\nTop 10 stocks by composite score:")
    cols = ["rank", "ticker", "sector",
            "score_valuation", "score_quality",
            "score_momentum",  "composite_score"]
    print(df[cols].head(10).to_string(index=False))

    return df


def save_scores(df: pd.DataFrame,
                path: str = "data/scores.csv") -> None:
    """Save the full scored DataFrame to CSV for inspection."""
    df.to_csv(path, index=False)
    print(f"\nScores saved to {path}")


if __name__ == "__main__":
    df = run_factor_engine()
    save_scores(df)