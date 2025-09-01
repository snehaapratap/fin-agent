import torch, torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=5, hidden=64, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        with torch.no_grad():
            q = self.forward(torch.tensor(obs).float().unsqueeze(0))
            return int(q.argmax(dim=1).item())
