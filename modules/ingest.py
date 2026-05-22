import yfinance as yf
import pandas as pd
import sqlite3
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from modules.universe import load_universe

DB_PATH = "data/equilens.db"
TABLE = "fundamentals"
BATCH_SIZE = 25       # fetch N tickers at once via yfinance download
SLEEP_BETWEEN = 2.0   # seconds to sleep between batches (rate limit safety)
MAX_AGE_DAYS = 3      # re-fetch if data is older than this many days


def get_connection():
    """Return a SQLite connection. Creates the DB file if it doesn't exist."""
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_table():
    """Create the fundamentals table if it doesn't already exist."""
    conn = get_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            ticker          TEXT PRIMARY KEY,
            market_cap      REAL,
            pe_ratio        REAL,
            ev_ebitda       REAL,
            pb_ratio        REAL,
            roe             REAL,
            roic            REAL,
            gross_margin    REAL,
            debt_to_equity  REAL,
            revenue_growth  REAL,
            earnings_growth REAL,
            price_1m        REAL,
            price_3m        REAL,
            price_6m        REAL,
            fetched_at      TEXT
        )
    """)
    conn.commit()
    conn.close()
def compute_price_return(ticker_obj, period: str) -> float:
    """
    Fetch price history and compute % return over the period.
    Returns None if not enough data.
    period: '1mo', '3mo', '6mo'
    """
    try:
        hist = ticker_obj.history(period=period)
        if len(hist) < 2:
            return None
        start_price = hist["Close"].iloc[0]
        end_price   = hist["Close"].iloc[-1]
        return round((end_price - start_price) / start_price, 6)
    except Exception:
        return None


def compute_roic(info: dict) -> float:
    """
    ROIC = Net Income / (Total Equity + Total Debt)
    yfinance doesn't expose this directly so we compute it from balance sheet fields.
    Returns None if data is missing.
    """
    try:
        net_income    = info.get("netIncomeToCommon")
        total_equity  = info.get("bookValue", 0) * info.get("sharesOutstanding", 0)
        total_debt    = info.get("totalDebt", 0)
        invested_cap  = total_equity + total_debt
        if not net_income or invested_cap == 0:
            return None
        return round(net_income / invested_cap, 6)
    except Exception:
        return None


def fetch_single(ticker: str) -> dict:
    """
    Fetch all fundamentals for one ticker.
    Returns a dict ready to INSERT into SQLite.
    Missing fields are stored as None (NULL) — not 0.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        # Guard: if yfinance returns an empty dict, skip this ticker
        if not info or info.get("regularMarketPrice") is None:
            return None

        row = {
            "ticker":          ticker,
            "market_cap":      info.get("marketCap"),
            "pe_ratio":        info.get("trailingPE"),
            "ev_ebitda":       info.get("enterpriseToEbitda"),
            "pb_ratio":        info.get("priceToBook"),
            "roe":             info.get("returnOnEquity"),
            "roic":            compute_roic(info),
            "gross_margin":    info.get("grossMargins"),
            "debt_to_equity":  info.get("debtToEquity"),
            "revenue_growth":  info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "price_1m":        compute_price_return(t, "1mo"),
            "price_3m":        compute_price_return(t, "3mo"),
            "price_6m":        compute_price_return(t, "6mo"),
            "fetched_at":      datetime.now().isoformat(),
        }
        return row
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None
def upsert_row(conn, row: dict):
    """Insert or replace a row in the fundamentals table."""
    cols = ", ".join(row.keys())
    vals = ", ".join(["?" for _ in row])
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE} ({cols}) VALUES ({vals})",
        list(row.values())
    )


def get_stale_tickers(all_tickers: list) -> list:
    """
    Return tickers that are either:
    - Not in the DB at all, OR
    - Have a fetched_at older than MAX_AGE_DAYS
    """
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=MAX_AGE_DAYS)).isoformat()

    existing = pd.read_sql(
        f"SELECT ticker, fetched_at FROM {TABLE}",
        conn
    )
    conn.close()

    fresh = set(
        existing[existing["fetched_at"] > cutoff]["ticker"].tolist()
    )
    stale = [t for t in all_tickers if t not in fresh]
    return stale


def run_ingestion(limit: int = None):
    """
    Main entry point. Fetches fundamentals for all stale tickers.
    limit: optionally cap how many tickers to fetch (useful for testing)
    """
    create_table()

    universe = load_universe()
    all_tickers = universe["ticker"].tolist()

    if limit:
        all_tickers = all_tickers[:limit]

    stale = get_stale_tickers(all_tickers)
    print(f"\nTotal tickers : {len(all_tickers)}")
    print(f"Already fresh : {len(all_tickers) - len(stale)}")
    print(f"To fetch      : {len(stale)}\n")

    if not stale:
        print("All data is fresh. Nothing to fetch.")
        return

    conn = get_connection()
    success, failed = 0, 0

    # Process in batches with a sleep between each to respect rate limits
    for i in tqdm(range(0, len(stale), BATCH_SIZE), desc="Fetching batches"):
        batch = stale[i : i + BATCH_SIZE]

        for ticker in batch:
            row = fetch_single(ticker)
            if row:
                upsert_row(conn, row)
                success += 1
            else:
                failed += 1

        conn.commit()

        # Sleep between batches — not between individual tickers
        if i + BATCH_SIZE < len(stale):
            time.sleep(SLEEP_BETWEEN)

    conn.close()
    print(f"\nDone. Success: {success}  |  Failed/skipped: {failed}")
    print(f"Database: {DB_PATH}")


def load_fundamentals(min_market_cap_B: float = 2.0) -> pd.DataFrame:
    """
    Load the full fundamentals table from SQLite.
    Called by Module 3 (Factor Engine).
    Filters out stocks below the market cap threshold.
    """
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()

    # Convert market cap from raw dollars to billions for the filter
    df = df[df["market_cap"] >= min_market_cap_B * 1e9]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} stocks from DB (market cap >= ${min_market_cap_B}B)")
    return df

if __name__ == "__main__":
    run_ingestion()   # removed limit=10 — runs all stale tickers