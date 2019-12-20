import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU(x)
        state_value = self.critic(x)
        action_scores = self.actor(x)
        return F.softmax(action_scores, dim=1), state_value

class AC():
    def __init__(self):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.log_probs = []
        self.V = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.V.append(value)
        return action.item()

    def update(self, rewards):
        policy_loss = []
        returns =  - torch.tensor(rewards)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.log_probs[:]
