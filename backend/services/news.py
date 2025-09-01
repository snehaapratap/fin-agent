import os, requests
from ..utils.cache import cache

NEWSAPI = "https://newsapi.org/v2/everything"
NEWS_KEY = os.getenv("NEWSAPI_API_KEY", "")

def search_news(query: str, days: int = 7):
    if not NEWS_KEY:
        return []
    key = f"news:{query}:{days}"
    c = cache.get(key)
    if c is not None:
        return c
    from datetime import datetime, timedelta
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "apiKey": NEWS_KEY,
        "language": "en"
    }
    r = requests.get(NEWSAPI, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("articles", [])
    cache.set(key, data)
    return data
