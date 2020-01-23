import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Policy, self).__init__()
        self.Regression = nn.Sequential(nn.Linear(state_dim, n_latent_var), nn.ReLU(), nn.Linear(n_latent_var, action_dim))

    def forward(self, x):
        action_scores = self.Regression(x)
        return F.softmax(action_scores, dim=1)

class Reinforce():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.gamma = gamma
        self.policy = Policy(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.log_probs = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def update(self, memory):
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        # Normalizing the rewards:
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.log_probs[:]
