import numpy as np
from crewai import Crew
from .agents import news_analyst, sentiment_agent, trader_agent, risk_manager_agent
from .tasks import news_task, sentiment_task, trader_task, risk_task
from ..services.market import daily_prices
from ..services.news import search_news
from ..services.sentiment import score_texts
from ..services.risk import RiskManager
from ..rl.train import train_policy

class AdvisorCrew:
    def __init__(self, symbols, capital=100000.0):
        self.symbols = symbols
        self.capital = capital

    def _collect_market(self):
        prices = {}
        for s in self.symbols:
            df = daily_prices(s)
            prices[s] = df[['date','adjusted close' if 'adjusted close' in df.columns else '5. adjusted close']].rename(columns=lambda x: x.replace('adjusted close','close').replace('5. adjusted close','close'))
        return prices

    def _collect_news_and_sentiment(self):
        raw = {}
        sent = {}
        for s in self.symbols:
            arts = search_news(s, days=7)
            raw[s] = arts
            texts = [a.get('title','')+". "+a.get('description','') for a in arts][:25]
            sent[s] = score_texts(texts) if texts else 0.0
        return raw, sent

    def run(self):
        prices = self._collect_market()
        raw_news, sent = self._collect_news_and_sentiment()

        policies = {}
        for s, df in prices.items():
            p = df['close'].values.astype(float)
            if len(p) > 40:
                policies[s] = train_policy(p, steps=1000)

        context = {
            "prices": {s: df.tail(60).to_dict(orient='records') for s, df in prices.items()},
            "news": raw_news,
            "sentiment": sent,
        }

        na, sa, ta, ra = news_analyst(), sentiment_agent(), trader_agent(), risk_manager_agent()
        crew = Crew(agents=[na, sa, ta, ra], tasks=[
            news_task(na, self.symbols, context),
            sentiment_task(sa, self.symbols, context),
            trader_task(ta, context),
            risk_task(ra, context),
        ])
        result = crew.kickoff()

        import json, re
        try:
            json_text = re.findall(r"\[.*\]", result, re.S)[-1]
            decisions = json.loads(json_text)
        except Exception:
            decisions = []

        rm = RiskManager(self.capital)
        for d in decisions:
            d['size'] = int(rm.position_size(float(prices[d['symbol']]['close'].iloc[-1]))) if 'size' not in d else d['size']
            if not rm.allow(d.get('action','hold')):
                d['action'] = 'hold'
        return {
            "insights": str(result)[:4000],
            "sentiment": sent,
            "news": {k: [{"title": a.get('title'), "url": a.get('url')} for a in v[:5]] for k, v in raw_news.items()},
            "decisions": decisions,
        }
