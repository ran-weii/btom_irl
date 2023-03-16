import torch
import torch.nn as nn
from src.tabular.planning import value_iteration

class DiscreteAgent(nn.Module):
    def __init__(self, state_dim, act_dim, gamma, alpha, horizon):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): action dimension
            gamma (float): discount factor
            alpha (float): softmax temperature
            horizon (int): finite planning horizon. Infinite horizon if horizon=0
        """
        super().__init__()
        self.finite_horizon = horizon != 0 # zero for infinite horizon
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.alpha = alpha
        self.horizon = horizon

        self.log_transition = nn.Parameter(torch.zeros(act_dim, state_dim, state_dim))
        self.log_target = nn.Parameter(torch.zeros(state_dim))
    
    def transition(self):
        return torch.softmax(self.log_transition, dim=-1)
    
    def reward(self):
        """ Compute log target distribution """
        return torch.log_softmax(self.log_target, dim=-1)

    def plan(self, bonus=None):
        with torch.no_grad():
            transition = self.transition()
            reward = self.reward()
            if bonus is not None:
                reward += bonus
            reward = reward.view(-1, 1).repeat_interleave(self.act_dim, -1)
        
        if self.finite_horizon:
            q, _ = value_iteration(
                transition, reward, self.gamma, self.alpha, 
                finite_horizon=True, max_iter=self.horizon
            )
        else:
            q, _ = value_iteration(
                transition, reward, self.gamma, self.alpha,
                finite_horizon=False
            )
            q = q[-1]
        v = torch.logsumexp(self.alpha * q, dim=-1) / self.alpha
        pi = torch.softmax(self.alpha * q, dim=-1)
        
        self.q = q
        self.v = v
        self.pi = pi

    def choose_action(self, s):
        if self.finite_horizon:
            pi = self.pi[-1][s.long()]
        else:
            pi = self.pi[s.long()]
        a = torch.multinomial(pi, 1).squeeze(-1)
        return a