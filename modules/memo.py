import pandas as pd
import numpy as np
import json
import os
from datetime import date
from jinja2 import Environment, FileSystemLoader

from modules.scorer     import load_watchlist
from modules.sentiment  import load_sentiment
from modules.divergence import load_divergence


def load_cfg() -> dict:
    with open("config.json") as f:
        return json.load(f)


def fmt(val, decimals=2, pct=False, fallback="N/A"):
    """Format a number cleanly. Returns fallback string for None/NaN."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return fallback
        if pct:
            return f"{val*100:.1f}%"
        return f"{val:.{decimals}f}"
    except Exception:
        return fallback


DIVERGENCE_LABELS = {
    "CONFIRMED_BUY":      "✓ Confirmed Buy — All Signals Aligned",
    "HIDDEN_GEM":         "◆ Hidden Gem — Fundamentals Ahead of Price",
    "MOMENTUM_RECOVERY":  "↑ Momentum Recovery — Fundamentals + Negative Momentum",
    "ALIGNED":            "— Aligned — No Significant Divergence",
    "SENTIMENT_RISK":     "⚠ Sentiment Risk — Management Tone Deteriorating",
    "MOMENTUM_TRAP":      "✗ Momentum Trap — Price Ahead of Fundamentals",
}


def generate_thesis(row: pd.Series) -> str:
    """
    Auto-generate a 2-3 sentence investment thesis from the stock's data.
    Uses templates keyed to divergence type + pillar scores.
    Not LLM-generated — deterministic from the data.
    """
    ticker  = row["ticker"]
    sector  = row.get("sector", "")
    val_z   = row.get("score_valuation",  0) or 0
    qual_z  = row.get("score_quality",    0) or 0
    mom_z   = row.get("score_momentum",   0) or 0
    comp    = row.get("composite_score",  0) or 0
    dtype   = row.get("divergence_type",  "ALIGNED")
    pe      = fmt(row.get("pe_ratio"),    1)
    roic    = fmt(row.get("roic"),        3, pct=True)
    gm      = fmt(row.get("gross_margin"),1, pct=True)
    p3m     = fmt(row.get("price_3m"),    1, pct=True)

    thesis_map = {
        "CONFIRMED_BUY": (
            f"{ticker} ranks in the top tier of the S&P 500 universe on a composite "
            f"of valuation (z={val_z:+.2f}), quality (z={qual_z:+.2f}), and momentum "
            f"(z={mom_z:+.2f}) factors. "
            f"With a P/E of {pe}x and ROIC of {roic}, the stock appears attractively "
            f"priced relative to its {sector} peers while demonstrating durable return "
            f"on invested capital. "
            f"Management sentiment is constructive and improving, with all three data "
            f"layers — quantitative factors, price momentum, and qualitative tone — "
            f"pointing in the same direction."
        ),
        "SENTIMENT_RISK": (
            f"{ticker} scores strongly on quantitative factors (composite z={comp:+.2f}), "
            f"with valuation z={val_z:+.2f} and quality z={qual_z:+.2f} both constructive "
            f"relative to {sector} peers. "
            f"However, the sentiment pipeline detected deteriorating management tone on "
            f"recent earnings calls — a divergence that warrants caution before "
            f"initiating a position. "
            f"The thesis remains intact on fundamentals, but investors should monitor "
            f"the next earnings call for confirmation that the tone decline is transitory "
            f"rather than structural."
        ),
        "MOMENTUM_TRAP": (
            f"{ticker} has delivered strong 3-month price performance ({p3m}) "
            f"that is not yet supported by fundamental factor scores "
            f"(val z={val_z:+.2f}, qual z={qual_z:+.2f}). "
            f"The composite score of {comp:+.2f} is driven primarily by momentum, "
            f"which tends to mean-revert when it outpaces underlying business quality. "
            f"This is a high-risk, high-monitoring name: compelling if fundamentals "
            f"catch up, dangerous if momentum reverses first."
        ),
        "HIDDEN_GEM": (
            f"{ticker} exhibits strong fundamental characteristics — valuation z={val_z:+.2f} "
            f"and quality z={qual_z:+.2f} — that the market has not yet recognised in "
            f"price momentum (z={mom_z:+.2f}). "
            f"With ROIC of {roic} and gross margin of {gm}, the underlying business "
            f"quality is evident. "
            f"The investment thesis is a potential re-rating catalyst: once momentum "
            f"inflects, this stock could move rapidly toward its fundamental fair value."
        ),
        "ALIGNED": (
            f"{ticker} ranks #{int(row.get('rank',0))} in the S&P 500 universe with a "
            f"composite score of {comp:+.2f}, reflecting broadly consistent signals "
            f"across valuation (z={val_z:+.2f}), quality (z={qual_z:+.2f}), and "
            f"momentum (z={mom_z:+.2f}) factors. "
            f"Operating in the {sector} sector with a P/E of {pe}x and ROIC of {roic}, "
            f"the risk/reward is consistent with the model's conviction level."
        ),
    }

    return thesis_map.get(dtype, thesis_map["ALIGNED"])


def build_risk_flags(row: pd.Series) -> list[dict]:
    """Build a list of risk flag dicts for the memo template."""
    flags = []

    de = row.get("debt_to_equity")
    if de and not np.isnan(de):
        if de > 200:
            flags.append({"style": "warn", "text": f"High leverage D/E {de:.0f}x"})
        elif de < 30:
            flags.append({"style": "ok",   "text": f"Low leverage D/E {de:.1f}x"})

    rev_g = row.get("revenue_growth")
    if rev_g and not np.isnan(rev_g):
        if rev_g < 0:
            flags.append({"style": "warn", "text": f"Revenue declining {rev_g*100:.1f}%"})
        elif rev_g > 0.3:
            flags.append({"style": "ok",   "text": f"Strong revenue growth {rev_g*100:.0f}%"})

    dtype = row.get("divergence_type", "")
    if dtype == "MOMENTUM_TRAP":
        flags.append({"style": "warn", "text": "Momentum trap — reversal risk"})
    if dtype == "SENTIMENT_RISK":
        flags.append({"style": "warn", "text": "Management tone deteriorating"})
    if dtype == "CONFIRMED_BUY":
        flags.append({"style": "ok",   "text": "All signals aligned"})
    if dtype == "HIDDEN_GEM":
        flags.append({"style": "info", "text": "Re-rating candidate"})

    tier = row.get("tier", "")
    if tier == "A":
        flags.append({"style": "ok",   "text": "Tier A — top 5% universe"})
    elif tier == "C":
        flags.append({"style": "info", "text": "Tier C — top 25% universe"})

    if not flags:
        flags.append({"style": "info", "text": "No major flags"})

    return flags
def build_memo_context(row: pd.Series) -> dict:
    """Build the full template context dict for one stock."""
    mc = row.get("market_cap")
    mc_b = fmt(mc / 1e9, 1) if mc and not np.isnan(mc) else "N/A"

    q1 = row.get("q1_score")
    qd = row.get("qoq_delta")

    return {
        # Identity
        "ticker":          row["ticker"],
        "company":         row.get("company", row["ticker"]),
        "sector":          row.get("sector", ""),
        "sub_industry":    row.get("sub_industry", ""),
        "rank":            int(row.get("rank", 0)),
        "tier":            row.get("tier", "—"),
        "confidence":      int(row.get("confidence", 0)) if not pd.isna(row.get("confidence", 0)) else "N/A",
        "generated_date":  date.today().strftime("%B %d, %Y"),
        # Scores
        "composite_score":  float(row.get("composite_score", 0) or 0),
        "score_valuation":  float(row.get("score_valuation",  0) or 0),
        "score_quality":    float(row.get("score_quality",    0) or 0),
        "score_momentum":   float(row.get("score_momentum",   0) or 0),
        # Raw metrics
        "pe_ratio":         fmt(row.get("pe_ratio"),         1),
        "ev_ebitda":        fmt(row.get("ev_ebitda"),        1),
        "pb_ratio":         fmt(row.get("pb_ratio"),         2),
        "roic":             fmt(row.get("roic"),             1, pct=True),
        "gross_margin":     fmt(row.get("gross_margin"),     1, pct=True),
        "debt_to_equity":   fmt(row.get("debt_to_equity"),   1),
        "price_1m":         fmt(row.get("price_1m"),         1, pct=True),
        "price_3m":         fmt(row.get("price_3m"),         1, pct=True),
        "price_6m":         fmt(row.get("price_6m"),         1, pct=True),
        "market_cap_b":     mc_b,
        # Sentiment
        "q1_score":         round(q1, 3) if q1 and not np.isnan(q1) else None,
        "qoq_delta":        round(qd, 3) if qd and not np.isnan(qd) else None,
        "tone_trend":       row.get("tone_trend",      "stable"),
        "sentiment_flag":   row.get("sentiment_flag",  "NEUTRAL"),
        # Divergence
        "divergence_type":        row.get("divergence_type", "ALIGNED"),
        "divergence_label":       DIVERGENCE_LABELS.get(
                                      row.get("divergence_type", "ALIGNED"),
                                      "— Aligned"
                                  ),
        "divergence_explanation": row.get("divergence_explanation", ""),
        # Generated content
        "thesis":      generate_thesis(row),
        "risk_flags":  build_risk_flags(row),
    }


def run_memo_generator() -> None:
    """
    Full Module 7 pipeline:
    1. Load watchlist (top 20 stocks from Module 4)
    2. Merge divergence data (Module 6)
    3. Render one HTML memo per stock via Jinja2
    4. Save to output/memos/{TICKER}.html
    """
    cfg      = load_cfg()
    out_dir  = cfg["output"]["memo_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 55)
    print("MODULE 7 — MEMO GENERATOR")
    print("=" * 55)

    # Load all data sources
    watchlist    = load_watchlist()
    divergence   = load_divergence()

    # Merge divergence onto watchlist
    div_cols = [
        "ticker", "divergence_type", "divergence_explanation",
        "q1_score", "qoq_delta", "tone_trend", "sentiment_flag"
    ]
    div_cols = [c for c in div_cols if c in divergence.columns]
    df = watchlist.merge(divergence[div_cols], on="ticker", how="left")

    # Fill defaults
    df["divergence_type"]        = df["divergence_type"].fillna("ALIGNED")
    df["divergence_explanation"] = df["divergence_explanation"].fillna("Signals broadly consistent.")
    df["sentiment_flag"]         = df["sentiment_flag"].fillna("NEUTRAL")
    df["tone_trend"]             = df["tone_trend"].fillna("stable")

    # Set up Jinja2 environment
    env      = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("memo.html")

    print(f"\nGenerating {len(df)} memos → {out_dir}/\n")

    for _, row in df.iterrows():
        ticker  = row["ticker"]
        context = build_memo_context(row)
        html    = template.render(**context)

        out_path = os.path.join(out_dir, f"{ticker}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        div_type = row.get("divergence_type", "ALIGNED")
        icons = {
            "CONFIRMED_BUY":    "✓",
            "SENTIMENT_RISK":   "⚠",
            "MOMENTUM_TRAP":    "✗",
            "HIDDEN_GEM":       "◆",
            "MOMENTUM_RECOVERY":"↑",
            "ALIGNED":          "—",
        }
        icon = icons.get(div_type, "—")
        print(f"  {icon} {ticker:<8}  Rank #{int(row['rank']):<4}  "
              f"Tier {row.get('tier','?')}  "
              f"{div_type}")

    print(f"\n{'='*55}")
    print(f"Done. {len(df)} memos written to {out_dir}/")
    print(f"Open any file in your browser:")
    print(f"  start {out_dir}\\MO.html")


if __name__ == "__main__":
    run_memo_generator()