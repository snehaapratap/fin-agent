import os, requests, pandas as pd, time
from . import news
from ..utils.cache import cache

ALPHA = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")

def _call(params):
    k = str(sorted(params.items()))
    c = cache.get(k)
    if c is not None:
        return c
    r = requests.get(ALPHA, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache.set(k, data)
    time.sleep(12)  # AV free tier rate limit
    return data

def daily_prices(symbol: str) -> pd.DataFrame:
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":symbol,"apikey":API_KEY}
    j = _call(params)
    ts = j.get("Time Series (Daily)", {})
    df = pd.DataFrame(ts).T.reset_index().rename(columns={"index":"date"})
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)
