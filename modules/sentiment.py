import requests
import json
import os
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

def load_cfg() -> dict:
    with open("config.json") as f:
        return json.load(f)["sentiment"]

EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
HEADERS = {
    "User-Agent": "EquiLens Research avish.vijay.shetty2026@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html",
}

_TICKER_CIK_MAP = {}

def load_cik_map() -> dict:
    """Load SEC's full ticker→CIK mapping once and cache in memory."""
    global _TICKER_CIK_MAP
    if _TICKER_CIK_MAP:
        return _TICKER_CIK_MAP
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        _TICKER_CIK_MAP[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return _TICKER_CIK_MAP


def get_cik(ticker: str) -> str | None:
    try:
        cik_map = load_cik_map()
        return cik_map.get(ticker.upper())
    except Exception as e:
        print(f"  [WARN] CIK lookup failed for {ticker}: {e}")
        return None


def fetch_transcripts(ticker: str,
                      api_key: str = None,
                      n: int = 2,
                      cache_dir: str = "data/transcripts") -> list[dict]:
    """
    Fetch n earnings call transcripts from SEC EDGAR.
    Uses primaryDocument field from submissions JSON — no index.json needed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        if cached:            # non-empty cache
            return cached[:n]
        # empty cache means previous run found nothing — try again
        os.remove(cache_file)

    cik = get_cik(ticker)
    if not cik:
        return []

    time.sleep(0.15)

    # ── Fetch the submissions JSON ──────────────────────────────────────────
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(sub_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] Submissions fetch failed for {ticker}: {e}")
        return []

    filings = data.get("filings", {}).get("recent", {})
    forms        = filings.get("form", [])
    accessions   = filings.get("accessionNumber", [])
    dates        = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    descriptions = filings.get("primaryDocDescription", [])

    results = []
    cik_nopad = str(int(cik))  # archive path uses non-padded CIK

    for form, acc, date, doc, desc in zip(
            forms, accessions, dates, primary_docs, descriptions):

        if len(results) >= n:
            break
        if form != "8-K":
            continue
        if not doc:
            continue

        # Build direct URL to the primary document
        acc_nodash = acc.replace("-", "")
        doc_url = f"{EDGAR_ARCHIVE}/{cik_nopad}/{acc_nodash}/{doc}"

        time.sleep(0.15)
        text = extract_text(doc_url)
        if not text:
            continue

        # Confirm it reads like an earnings call transcript
        call_keywords = ["earnings", "revenue", "quarter", "operator",
                         "questions", "analyst", "guidance", "per share"]
        if sum(1 for kw in call_keywords if kw in text.lower()) < 3:
            continue

        results.append({
            "date":    date,
            "content": text,
            "source":  "SEC EDGAR",
            "url":     doc_url,
        })

    # Cache result (even if empty, to avoid re-hitting EDGAR)
    with open(cache_file, "w") as f:
        json.dump(results, f)

    return results[:n]


def extract_text(url: str) -> str | None:
    """Fetch an EDGAR filing and return clean plain text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        # Handle both HTML and plain text filings
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or url.endswith((".htm", ".html")):
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        else:
            text = resp.text

        text = re.sub(r"\s+", " ", text).strip()

        # Must be substantive content
        if len(text.split()) < 500:
            return None
        return text

    except Exception as e:
        return None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Finance-specific word overrides for VADER
# VADER was trained on social media — these words have different valence in finance
FINANCE_LEXICON_OVERRIDES = {
    "beat":        2.5,   # beat estimates
    "miss":       -2.5,   # missed estimates
    "raised":      1.5,   # raised guidance
    "lowered":    -1.5,   # lowered guidance
    "headwinds":  -1.8,
    "tailwinds":   1.8,
    "challenging": -1.2,
    "uncertainty": -1.0,
    "confident":   1.5,
    "cautious":   -1.0,
    "accelerating":1.5,
    "decelerating":-1.5,
    "robust":      1.5,
    "weakness":   -1.5,
    "strength":    1.2,
    "margin":      0.0,   # neutral — contextual
    "pressure":   -1.0,
    "momentum":    1.0,
    "headcount":  -0.5,   # often precedes reduction language
}

_vader = None  # lazy-load

def get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
        # Apply finance overrides
        _vader.lexicon.update(FINANCE_LEXICON_OVERRIDES)
    return _vader


def score_text_vader(text: str) -> float:
    """
    Score a block of text sentence-by-sentence using VADER.
    Returns the mean compound score across all sentences.
    Compound score range: -1.0 (most negative) to +1.0 (most positive).
    """
    analyser = get_vader()

    # Split into sentences (naive but sufficient for earnings transcripts)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]

    if not sentences:
        return 0.0

    scores = [analyser.polarity_scores(s)["compound"] for s in sentences]
    return round(float(np.mean(scores)), 4)
_finbert_pipeline = None  # lazy-load — only downloaded if model=finbert

def get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        cfg = load_cfg()
        print("  Loading FinBERT model (first run: ~440MB download)...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model=cfg["finbert_model"],
            tokenizer=cfg["finbert_model"],
            truncation=True,
            max_length=512,
        )
        print("  FinBERT loaded.")
    return _finbert_pipeline


def score_text_finbert(text: str, batch_size: int = 16) -> float:
    """
    Score text using FinBERT.
    FinBERT outputs: positive / negative / neutral with a probability score.
    We convert to a signed float:
        positive → +probability
        negative → -probability
        neutral  → 0

    Returns mean signed score across all sentences.
    """
    pipe = get_finbert()

    # Chunk into sentences, max 400 chars each (FinBERT 512-token limit)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]
    sentences = [s[:400] for s in sentences]

    if not sentences:
        return 0.0

    results = pipe(sentences, batch_size=batch_size)

    signed_scores = []
    for r in results:
        label = r["label"].lower()
        prob  = r["score"]
        if label == "positive":
            signed_scores.append(prob)
        elif label == "negative":
            signed_scores.append(-prob)
        else:
            signed_scores.append(0.0)

    return round(float(np.mean(signed_scores)), 4)
def score_transcript(transcript: dict, model: str = "vader") -> float:
    """Score a single transcript dict returned by fetch_transcripts()."""
    content = transcript.get("content", "")
    if not content or len(content) < 100:
        return None
    if model == "finbert":
        return score_text_finbert(content)
    return score_text_vader(content)


def analyse_ticker(ticker: str,
                   api_key: str,
                   n: int = 2,
                   model: str = "vader",
                   cache_dir: str = "data/transcripts",
                   sleep: float = 0.4) -> dict:
    """
    Full pipeline for one ticker:
    1. Fetch last n transcripts
    2. Score each one
    3. Compute QoQ delta (score[0] - score[1], i.e. latest minus previous)
    4. Assign a tone trend label

    Returns a dict with all results.
    """
    transcripts = fetch_transcripts(ticker, api_key, n, cache_dir)
    time.sleep(sleep)  # rate limit: 250 calls/day free tier

    if not transcripts:
        return {
            "ticker": ticker,
            "q1_date": None, "q1_score": None,
            "q2_date": None, "q2_score": None,
            "qoq_delta": None,
            "tone_trend": "no_data",
            "sentiment_flag": None,
        }

    scores = []
    dates  = []
    for t in transcripts:
        s = score_transcript(t, model)
        scores.append(s)
        dates.append(t.get("date", ""))

    # scores[0] = most recent, scores[1] = one quarter ago
    q1 = scores[0] if len(scores) > 0 else None
    q2 = scores[1] if len(scores) > 1 else None

    # QoQ delta: positive = tone improving, negative = tone deteriorating
    if q1 is not None and q2 is not None:
        delta = round(q1 - q2, 4)
    else:
        delta = None

    # Tone trend label
    if delta is None:
        trend = "insufficient_data"
    elif delta >= 0.05:
        trend = "improving"
    elif delta <= -0.05:
        trend = "deteriorating"
    else:
        trend = "stable"

    # Sentiment flag for memo generator
    # Combines absolute level + trend into an actionable signal
    if q1 is not None and q1 >= 0.10 and trend == "improving":
        flag = "POSITIVE"
    elif q1 is not None and q1 <= -0.05:
        flag = "NEGATIVE"
    elif trend == "deteriorating":
        flag = "CAUTION"
    else:
        flag = "NEUTRAL"

    return {
        "ticker":       ticker,
        "q1_date":      dates[0] if dates else None,
        "q1_score":     q1,
        "q2_date":      dates[1] if len(dates) > 1 else None,
        "q2_score":     q2,
        "qoq_delta":    delta,
        "tone_trend":   trend,
        "sentiment_flag": flag,
    }


def run_sentiment(save: bool = True) -> pd.DataFrame:
    """
    Full Module 5 pipeline:
    1. Load top-N tickers from scores.csv
    2. Fetch + score transcripts for each
    3. Save sentiment.csv

    Returns DataFrame with one row per ticker.
    """
    cfg = load_cfg()
    api_key   = cfg["fmp_api_key"]
    top_n     = cfg["top_n_stocks"]
    n_trans   = cfg["transcripts_per_stock"]
    model     = cfg["model"]
    cache_dir = cfg["transcripts_dir"]
    out_path  = cfg["output_path"]

    if api_key == "YOUR_KEY_HERE":
        raise ValueError("Set your FMP API key in config.json → sentiment.fmp_api_key")

    # Load top-N from scores.csv (Module 3/4 output)
    scores_path = "data/scores.csv"
    if not os.path.exists(scores_path):
        raise FileNotFoundError("Run modules/scorer.py first to generate data/scores.csv")

    scores_df = pd.read_csv(scores_path)
    top_tickers = scores_df.sort_values("rank").head(top_n)["ticker"].tolist()

    print("=" * 55)
    print(f"MODULE 5 — SENTIMENT PIPELINE ({model.upper()})")
    print("=" * 55)
    print(f"Scoring top {top_n} tickers from scores.csv\n")

    results = []
    for i, ticker in enumerate(top_tickers):
        row = analyse_ticker(ticker, api_key, n_trans, model, cache_dir)
        results.append(row)
        flag = row["sentiment_flag"] or "—"
        delta_str = f"{row['qoq_delta']:+.3f}" if row["qoq_delta"] is not None else "N/A"
        print(f"  [{i+1:02d}/{top_n}] {ticker:<6}  "
              f"Q1={row['q1_score'] or 'N/A':>6}  "
              f"Δ={delta_str:>7}  "
              f"{row['tone_trend']:<18} → {flag}")

    df = pd.DataFrame(results)

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSentiment saved to {out_path}")

    # Summary
    print(f"\n── Sentiment flag summary ──")
    print(df["sentiment_flag"].value_counts(dropna=False).to_string())
    print(f"\n── Tone trend summary ──")
    print(df["tone_trend"].value_counts(dropna=False).to_string())

    return df


# ── Utility: load cached sentiment (used by Module 7) ────────────────────────
def load_sentiment() -> pd.DataFrame:
    cfg = load_cfg()
    path = cfg["output_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sentiment not found at {path}. Run sentiment.py first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    run_sentiment()