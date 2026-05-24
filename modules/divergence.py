import pandas as pd
import numpy as np
import json
import os

# ── Divergence thresholds ─────────────────────────────────────────────────────
# These define how large a gap must be before we flag a divergence.
# Tuned to catch meaningful mismatches without over-flagging noise.

VAL_MOM_DIVERGENCE_THRESHOLD   = 1.5   # val_z - mom_z gap
QUAL_MOM_DIVERGENCE_THRESHOLD  = 1.5   # qual_z - mom_z gap
COMPOSITE_SENTIMENT_THRESHOLD  = 0.8   # high composite score cutoff
MOMENTUM_TRAP_MOM_THRESHOLD    = 1.2   # strong momentum z-score
MOMENTUM_TRAP_FUND_THRESHOLD   = -0.3  # weak fundamentals (val+qual avg)


def load_scores() -> pd.DataFrame:
    path = "data/scores.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run modules/scorer.py first.")
    return pd.read_csv(path)


def load_sentiment() -> pd.DataFrame:
    path = "data/sentiment.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run modules/sentiment.py first.")
    return pd.read_csv(path)


def classify_divergence(row: pd.Series) -> tuple[str, str]:
    """
    Classify a stock into a divergence type based on its factor scores
    and sentiment flag.

    Returns (divergence_type, explanation) tuple.

    Types:
      CONFIRMED_BUY    — composite strong + sentiment improving/positive
      SENTIMENT_RISK   — composite strong + sentiment deteriorating/negative
      HIDDEN_GEM       — strong val+qual + weak momentum (re-rating candidate)
      MOMENTUM_TRAP    — strong momentum + weak fundamentals
      MOMENTUM_RECOVERY— negative momentum + strong fundamentals (recovery play)
      ALIGNED          — no significant divergence, signals consistent
    """
    val_z    = row.get("score_valuation",  0) or 0
    qual_z   = row.get("score_quality",    0) or 0
    mom_z    = row.get("score_momentum",   0) or 0
    composite= row.get("composite_score",  0) or 0
    sent_flag= row.get("sentiment_flag",   "NEUTRAL") or "NEUTRAL"
    tone     = row.get("tone_trend",       "stable")  or "stable"

    fund_avg = (val_z + qual_z) / 2  # fundamental strength proxy

    # ── Type 4: Confirmed buy ────────────────────────────────────────────────
    if (composite >= COMPOSITE_SENTIMENT_THRESHOLD
            and sent_flag == "POSITIVE"
            and tone in ("improving", "stable")):
        return ("CONFIRMED_BUY",
                f"Strong composite ({composite:.2f}) + positive/improving sentiment. "
                f"All signals aligned — highest conviction.")

    # ── Type 2: Sentiment risk ───────────────────────────────────────────────
    if (composite >= COMPOSITE_SENTIMENT_THRESHOLD
            and sent_flag in ("CAUTION", "NEGATIVE")):
        return ("SENTIMENT_RISK",
                f"High composite ({composite:.2f}) but {sent_flag} sentiment "
                f"({tone}). Factor model is bullish; management tone is not. "
                f"Investigate before including in watchlist.")

    # ── Type 3: Momentum trap ────────────────────────────────────────────────
    if (mom_z >= MOMENTUM_TRAP_MOM_THRESHOLD
            and fund_avg < MOMENTUM_TRAP_FUND_THRESHOLD):
        return ("MOMENTUM_TRAP",
                f"Strong momentum (mom_z={mom_z:.2f}) but weak fundamentals "
                f"(val={val_z:.2f}, qual={qual_z:.2f}). "
                f"Price running ahead of fundamentals — high reversal risk.")

    # ── Type 1: Hidden gem ───────────────────────────────────────────────────
    if (fund_avg >= 0.8
            and mom_z <= -0.5):
        return ("HIDDEN_GEM",
                f"Strong fundamentals (val={val_z:.2f}, qual={qual_z:.2f}) "
                f"but negative momentum (mom_z={mom_z:.2f}). "
                f"Potential re-rating opportunity — market hasn't priced quality yet.")

    # ── Momentum recovery ────────────────────────────────────────────────────
    if (fund_avg >= 0.5
            and mom_z < -0.8):
        return ("MOMENTUM_RECOVERY",
                f"Good fundamentals + deeply negative momentum. "
                f"Could be a value trap or a recovery setup — "
                f"check sentiment for confirmation.")

    # ── Aligned ─────────────────────────────────────────────────────────────
    return ("ALIGNED",
            f"No significant divergence. Signals are broadly consistent.")
def run_divergence(save: bool = True) -> pd.DataFrame:
    """
    Full Module 6 pipeline:
    1. Load scores (all 503 stocks) and sentiment (top 50)
    2. Left-join sentiment onto scores
    3. Apply classify_divergence() to every row
    4. Print summary by divergence type
    5. Save divergence.csv

    Returns the full scored + classified DataFrame.
    """
    print("=" * 55)
    print("MODULE 6 — DIVERGENCE DETECTOR")
    print("=" * 55)

    scores_df    = load_scores()
    sentiment_df = load_sentiment()

    # Merge sentiment onto scores (left join — stocks with no sentiment
    # get NaN for sentiment columns, treated as NEUTRAL in classify)
    sentiment_cols = [
        "ticker", "q1_score", "qoq_delta", "tone_trend", "sentiment_flag"
    ]
    # Only keep sentiment cols that exist (robustness)
    sentiment_cols = [c for c in sentiment_cols if c in sentiment_df.columns]
    df = scores_df.merge(
        sentiment_df[sentiment_cols],
        on="ticker",
        how="left"
    )

    print(f"\nStocks in universe  : {len(scores_df)}")
    print(f"With sentiment data : {df['sentiment_flag'].notna().sum()}")
    print(f"Without sentiment   : {df['sentiment_flag'].isna().sum()}")

    # Fill NaN sentiment with NEUTRAL defaults so classify works on all stocks
    df["sentiment_flag"] = df["sentiment_flag"].fillna("NEUTRAL")
    df["tone_trend"]     = df["tone_trend"].fillna("stable")
    df["q1_score"]       = df["q1_score"].fillna(np.nan)

    # Apply divergence classification to every stock
    print("\nClassifying divergences...")
    classifications = df.apply(classify_divergence, axis=1)
    df["divergence_type"]        = classifications.apply(lambda x: x[0])
    df["divergence_explanation"] = classifications.apply(lambda x: x[1])

    # ── Print results by type ─────────────────────────────────────────────────
    type_order = [
        "CONFIRMED_BUY", "HIDDEN_GEM", "MOMENTUM_RECOVERY",
        "ALIGNED", "SENTIMENT_RISK", "MOMENTUM_TRAP"
    ]

    print()
    for dtype in type_order:
        subset = df[df["divergence_type"] == dtype]
        if subset.empty:
            continue

        icons = {
            "CONFIRMED_BUY":     "✓ CONFIRMED BUY",
            "HIDDEN_GEM":        "◆ HIDDEN GEM",
            "MOMENTUM_RECOVERY": "↑ MOMENTUM RECOVERY",
            "ALIGNED":           "— ALIGNED",
            "SENTIMENT_RISK":    "⚠ SENTIMENT RISK",
            "MOMENTUM_TRAP":     "✗ MOMENTUM TRAP",
        }
        print(f"\n{icons.get(dtype, dtype)} ({len(subset)} stocks)")
        print("-" * 55)

        display_cols = [
            "rank", "ticker", "sector",
            "composite_score", "score_valuation",
            "score_quality", "score_momentum",
            "sentiment_flag", "tone_trend"
        ]
        display_cols = [c for c in display_cols if c in subset.columns]

        # Only print top-30 for ALIGNED (there will be many)
        show = subset if dtype != "ALIGNED" else subset.head(30)
        print(show[display_cols].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        out_path = "data/divergence.csv"
        df.to_csv(out_path, index=False)
        print(f"\nDivergence report saved to {out_path}")

    # ── Summary counts ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    counts = df["divergence_type"].value_counts()
    for dtype in type_order:
        n = counts.get(dtype, 0)
        bar = "█" * min(n, 40)
        print(f"  {dtype:<22} {bar} {n}")

    return df


# ── Utility: load divergence report (used by Module 7) ───────────────────────
def load_divergence() -> pd.DataFrame:
    path = "data/divergence.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run modules/divergence.py first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = run_divergence()
