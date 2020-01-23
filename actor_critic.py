import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, n_latent_var)
        self.actor = nn.Linear(n_latent_var, action_dim)
        self.critic = nn.Linear(n_latent_var, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        action_scores = self.actor(out)
        value = self.critic(out)

        return F.softmax(action_scores, dim=1), value

class A2C():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.gamma = gamma
        self.policy = Policy(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.vs = []
        self.logprobs = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))
        self.vs.append(value)
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

        adv = returns - torch.tensor(self.vs)

        policy_loss = []
        for log_prob, A in zip(self.logprobs, adv):
            policy_loss.append(-log_prob * A)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() + torch.mean(adv*adv).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.logprobs[:]
        del self.vs[:]
