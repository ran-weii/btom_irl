import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agents.utils import (
    finite_horizon_riccati_equation, infinite_horizon_riccati_equation)

class LQRAgent(nn.Module):
    """ Linear quadratic control agent """
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
        
        self._A = nn.Parameter(torch.eye(state_dim))
        self._B = nn.Parameter(torch.zeros(state_dim, act_dim))
        self.log_I = nn.Parameter(torch.zeros(state_dim)) # log diagonal covariance
        self.log_Q = nn.Parameter(torch.zeros(state_dim))
        self.log_R = nn.Parameter(torch.zeros(act_dim))
    
    def A(self):
        """ Triu constrained transition matrix """
        return torch.triu(self._A)

    def B(self):
        """ Control matrix """
        return self._B

    def I(self):
        """ Diagonal noise covariance matrix """
        return torch.diag(self.log_I.exp() ** 2)
    
    def Q(self):
        """ Diagonal state cost matrix """
        return torch.diag(self.log_Q.exp())

    def R(self):
        """ Diagonal control cost matrix """
        return torch.diag(self.log_R.exp())

    def plan(self):
        A = self.A()
        B = self.B()
        I = self.I()
        Q = self.Q()
        R = self.R()
        
        if self.finite_horizon:
            Q_t, c_t = finite_horizon_riccati_equation(
                A, B, I, Q, R, 
                self.gamma, self.alpha, self.horizon
            )
        else:
            Q_t, c_t, _ = infinite_horizon_riccati_equation(
                A, B, I, Q, R, 
                self.gamma, self.alpha
            )
        
        self.Q_t = Q_t
        self.c_t = c_t
        self.K, self.Sigma = self.compute_action_params(Q_t)

    def compute_action_params(self, Q_t):
        """ Compute action distribution gain and covariance parameters
        
        Args:
            Q_t (torch.tensor): value quadratic matrices. size=[..., state_dim, state_dim]

        Returns:
            K (torch.tensor): action feedback gain. size=[state_dim, act_dim]
            Sigma (torch.tensor): action covariance. size=[act_dim, act_dim]
        """
        if self.finite_horizon:
            Q_t_ = Q_t[-1]
        else:
            Q_t_ = Q_t
        
        A = self.A()
        B = self.B()
        R = self.R()
        bqa = B.T.matmul(Q_t_).matmul(A)
        rbqb = R + self.gamma * B.T.matmul(Q_t_).matmul(B)
        rbqb_inv = torch.linalg.inv(rbqb)

        K = -self.gamma * rbqb_inv.matmul(bqa)
        Sigma = self.alpha * rbqb_inv
        return K, Sigma
    
    def compute_action_dist(self, s, K, Sigma):
        mu = s.matmul(K.T)
        pi = torch_dist.MultivariateNormal(mu, covariance_matrix=Sigma)
        return pi
    
    def compute_cost(self, s, a, Q, R):
        c_s = torch.einsum("...i, ij, ...j -> ...", s, Q, s)
        c_a = torch.einsum("...i, ij, ...j -> ...", a, R, a)
        c = 0.5 * (c_s + c_a)
        return c

    def compute_value(self, s, Q_t, c_t):
        v = 0.5 * torch.einsum("...i, ij, ...j -> ...", s, Q_t, s) + c_t
        return v

    def choose_action(self, s):
        pi = self.compute_action_dist(s, self.K, self.Sigma)
        a = pi.rsample()
        return a