import torch
import torch.nn as nn
from src.agents.utils import value_iteration

class DiscreteAgent(nn.Module):
    def __init__(self, state_dim, act_dim, gamma, alpha):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.alpha = alpha

        self.log_transition = nn.Parameter(torch.randn(act_dim, state_dim, state_dim))
        self.log_target = nn.Parameter(torch.randn(state_dim))

    def init_params(self, log_transition=None, log_target=None):
        if log_transition is not None:
            self.log_transition.data = log_transition

        if log_target is not None:
            self.log_target.data = log_target
    
    def transition(self):
        return torch.softmax(self.log_transition, dim=-1)
    
    def reward(self):
        """ Compute log target distribution """
        return torch.log_softmax(self.log_target, dim=-1)

    def plan(self):
        with torch.no_grad():
            transition = self.transition()
            reward = self.reward().view(-1, 1)

        q, _ = value_iteration(transition, reward, self.gamma, self.alpha)
        v = torch.logsumexp(self.alpha * q, dim=-1) / self.alpha
        pi = torch.softmax(self.alpha * q, dim=-1)
        
        self.q = q
        self.v = v
        self.pi = pi

    def choose_action(self, s):
        pi = self.pi[s]
        a = torch.multinomial(pi, 1)
        return a.numpy().flatten()