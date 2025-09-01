import os
MAX_POS_PCT = float(os.getenv("RISK_MAX_POSITION_PCT", 0.2))
MAX_DD = float(os.getenv("RISK_MAX_PORTFOLIO_DRAWDOWN", 0.25))

class RiskManager:
    def __init__(self, capital: float):
        self.capital = capital

    def position_size(self, price: float) -> int:
        budget = self.capital * MAX_POS_PCT
        return max(0, int(budget // max(price, 1e-6)))

    def allow(self, action: str) -> bool:
        return action in {"buy","sell","hold"}
