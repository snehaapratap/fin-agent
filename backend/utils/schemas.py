from pydantic import BaseModel
from typing import List, Optional


class AnalyzeRequest(BaseModel):
    symbols: List[str]
    lookback_days: int = 7


class AnalyzeResponse(BaseModel):
    insights: str
    sentiment: dict
    news: List[dict]


class TradeRequest(BaseModel):
    symbols: List[str]
    capital: float = 100000.0
    mode: str = "paper"  # or "backtest"


class TradeDecision(BaseModel):
    symbol: str
    action: str  # buy/sell/hold
    size: float
    reason: str


class TradeResponse(BaseModel):
    decisions: List[TradeDecision]
    notes: Optional[str] = None
