import os

import pandas as pd
import yfinance as yf

SECTOR_TICKERS = {
    "XLK": ["NVDA", "AAPL", "MSFT", "AVGO", "AMD", "MU", "INTC", "CSCO", "AMAT",
            "LRCX", "TXN", "KLAC", "ORCL", "IBM", "APH", "MRVL", "ADI", "STX", "QCOM", "CRM"],
    "XLY": ["AMZN", "HD", "MCD", "BKNG", "TJX", "SBUX", "LOW", "ROST", "MAR", "RCL",
            "ORLY", "F", "GRMN", "AZO", "NKE", "EBAY", "CMG", "YUM", "DHI"],
    "XLI": ["CAT", "GE", "RTX", "BA", "UNP", "ETN", "DE", "PH", "LMT", "ADP", "TT",
            "PWR", "GD", "MMM", "CSX", "JCI", "EMR", "CMI", "WM", "UPS", "NOC"],
    "XLF": ["JPM", "BRK-B", "V", "MA", "BAC", "GS", "WFC", "MS", "C", "AXP", "SCHW",
            "BLK", "COF", "SPGI", "CB", "PGR", "BNY", "BX", "PNC", "USB", "CME", "ICE", "TRV"],
    "XLE": ["XOM", "CVX", "COP", "VLO", "SLB", "EOG", "WMB", "BKR", "OKE", "DVN",
            "OXY", "EQT", "HAL", "TPL", "APA"],
}
TICKERS = [ticker for tickers in SECTOR_TICKERS.values() for ticker in tickers]
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

CACHE_PATH = "av_price_cache.csv"


def download_and_cache(tickers=TICKERS, cache_path=CACHE_PATH) -> pd.DataFrame:
    print(f"Downloading full daily history for {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(tickers, period="max", auto_adjust=True, progress=False)
    prices = data["Close"][tickers]
    prices.to_csv(cache_path)
    print(f"Cached raw prices to {cache_path}")
    return prices


def load_prices(tickers=TICKERS, cache_path=CACHE_PATH, force_refresh=False) -> pd.DataFrame:
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached prices from {cache_path}...")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            print(f"Cache missing {missing}, re-downloading full set...")
            prices = download_and_cache(tickers, cache_path)
    else:
        prices = download_and_cache(tickers, cache_path)

    return prices[tickers]


def load_returns(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    cache_path=CACHE_PATH,
    force_refresh=False,
    standardize=False,
):

    prices = load_prices(tickers, cache_path, force_refresh)

    prices = prices.loc[start_date:end_date]
    prices = prices.ffill()

    returns = prices.pct_change().dropna()

    if standardize:
        returns = (returns - returns.mean()) / returns.std()

    return returns, prices


if __name__ == "__main__":
    returns_df, prices_df = load_returns()
    print("\nReturns shape:", returns_df.shape)
    print("Date range:", returns_df.index.min().date(), "to", returns_df.index.max().date())
    print("\nHead:")
    print(returns_df.head())
    print("\nMissing values per column:")
    print(returns_df.isna().sum())
