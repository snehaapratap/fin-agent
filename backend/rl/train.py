import numpy as np
import torch, torch.optim as optim
from .env import TradingEnv
from .policy import MLPPolicy


def train_policy(prices_np: np.ndarray, steps=2000):
    env = TradingEnv(prices_np)
    net = MLPPolicy()
    tgt = MLPPolicy(); tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    gamma = 0.99
    buf_s, buf_a, buf_r, buf_ns, buf_d = [], [], [], [], []

    obs = env.reset()
    for t in range(steps):
        eps = max(0.05, 1.0 - t/1500)
        a = np.random.randint(0,3) if np.random.rand() < eps else net.act(obs)
        nobs, r, done, _ = env.step(a)
        buf_s.append(obs); buf_a.append(a); buf_r.append(r); buf_ns.append(nobs); buf_d.append(done)
        obs = nobs if not done else env.reset()
        if len(buf_s) >= 64:
            idx = np.random.choice(len(buf_s), 64, replace=False)
            s = torch.tensor(np.array([buf_s[i] for i in idx])).float()
            a = torch.tensor(np.array([buf_a[i] for i in idx])).long()
            r = torch.tensor(np.array([buf_r[i] for i in idx])).float()
            ns = torch.tensor(np.array([buf_ns[i] for i in idx])).float()
            d = torch.tensor(np.array([buf_d[i] for i in idx])).float()

            q = net(s).gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                qn = tgt(ns).max(1).values
                y = r + gamma * (1 - d) * qn
            loss = (q - y).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if t % 200 == 0:
            tgt.load_state_dict(net.state_dict())
    return net
