import pandas as pd
import requests
import os
from io import StringIO

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = "data/universe.csv"

MIN_MARKET_CAP_B = 2


def fetch_sp500() -> pd.DataFrame:
    """
    Scrape S&P 500 constituents from Wikipedia
    and return a cleaned DataFrame.
    """

    print("Fetching S&P 500 table from Wikipedia...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(WIKI_URL, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page: {response.status_code}")

    # Parse HTML tables
    tables = pd.read_html(StringIO(response.text))

    # Main S&P500 table
    df = tables[0]

    print(f"Raw columns found: {list(df.columns)}")

    # Rename columns
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Headquarters Location": "headquarters",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    })

    # Keep required columns
    keep = ["ticker", "company", "sector", "sub_industry"]

    df = df[[c for c in keep if c in df.columns]]

    # Fix ticker symbols
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    # Remove missing tickers
    df = df.dropna(subset=["ticker"])

    # Reset index
    df = df.reset_index(drop=True)

    print(
        f"Universe size: {len(df)} stocks across "
        f"{df['sector'].nunique()} sectors"
    )

    return df


def save_universe(df: pd.DataFrame) -> None:
    """
    Save universe to CSV.
    """

    os.makedirs("data", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved universe to: {OUTPUT_PATH}")


def load_universe() -> pd.DataFrame:
    """
    Load cached universe if available,
    otherwise fetch and cache it.
    """

    if os.path.exists(OUTPUT_PATH):
        print(f"Loading cached universe: {OUTPUT_PATH}")
        return pd.read_csv(OUTPUT_PATH)

    print("No cache found. Fetching fresh data...")

    df = fetch_sp500()

    save_universe(df)

    return df


def get_tickers_by_sector(sector: str) -> list:
    """
    Return all tickers for a given sector.
    """

    df = load_universe()

    return df[df["sector"] == sector]["ticker"].tolist()


def get_sector_map() -> dict:
    """
    Return mapping:
    { ticker -> sector }
    """

    df = load_universe()

    return dict(zip(df["ticker"], df["sector"]))


if __name__ == "__main__":

    df = fetch_sp500()

    # SAVE CSV HERE
    save_universe(df)

    print("\nSector breakdown:")
    print(df["sector"].value_counts().to_string())

    print("\nSample rows:")
    print(df.head(10).to_string(index=False))