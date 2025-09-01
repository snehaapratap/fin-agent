import gym
import numpy as np

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices):
        super().__init__()
        self.prices = prices.astype(np.float32)
        self.n = len(prices)
        self.i = 0
        self.holding = 0
        self.cash = 100000.0
        self.action_space = gym.spaces.Discrete(3)  # 0:hold,1:buy,2:sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _obs(self):
        window = self.prices[max(0,self.i-5):self.i]
        r = np.diff(window) / (window[:-1] + 1e-6) if len(window) > 1 else np.zeros(4)
        pad = np.zeros(4 - len(r))
        feat = np.concatenate([r, pad])
        pos = np.array([self.holding], dtype=np.float32)
        return np.concatenate([feat, pos]).astype(np.float32)

    def step(self, action):
        price = self.prices[self.i]
        reward = 0.0
        if action == 1 and self.cash >= price:
            self.holding += 1
            self.cash -= price
        elif action == 2 and self.holding > 0:
            self.holding -= 1
            self.cash += price
            reward = 0.0  # realized PnL recognized via cash
        self.i += 1
        done = self.i >= self.n-1
        next_price = self.prices[self.i]
        equity = self.cash + self.holding * next_price
        reward += equity / 100000.0  # scaled equity as shaping
        return self._obs(), float(reward), done, {}

    def reset(self):
        self.i = 0
        self.holding = 0
        self.cash = 100000.0
        return self._obs()
