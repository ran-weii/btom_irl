import torch
import torch.nn as nn
from src.agents.nn_models import MLP

class DoubleQNetwork(nn.Module):
    """ Double Q network for continuous actions """
    def __init__(self, obs_dim, act_dim, hidden_dim, num_hidden, activation="silu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.q1 = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )
        self.q2 = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )
    
    def __repr__(self):
        s = "{}(input_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.obs_dim + self.act_dim, self.q1.hidden_dim, 
            self.q1.num_hidden, self.q1.activation
        )
        return s

    def forward(self, o, a):
        """ Compute q1 and q2 values
        
        Args:
            o (torch.tensor): observation. size=[batch_size, obs_dim]
            a (torch.tensor): action. size=[batch_size, act_dim]

        Returns:
            q1 (torch.tensor): q1 value. size=[batch_size, 1]
            q2 (torch.tensor): q2 value. size=[batch_size, 1]
        """
        oa = torch.cat([o, a], dim=-1)
        q1 = self.q1(oa)
        q2 = self.q2(oa)
        return q1, q2


def compute_q_target(r, v_next, done, gamma, use_terminal=False):
    q_target = r + (1 - done) * gamma * v_next
    if use_terminal: # special handle terminal state
        v_done = gamma / (1 - gamma) * r
        q_target += done * v_done
    return q_target

def compute_critic_loss(
    batch, policy, critic, critic_target, gamma, beta, device, rwd_fn=None, use_terminal=False
    ):
    obs = batch["obs"].to(device)
    act = batch["act"].to(device)
    r = batch["rwd"].to(device)
    next_obs = batch["next_obs"].to(device)
    done = batch["done"].to(device)

    with torch.no_grad():
        if rwd_fn is not None:
            r = rwd_fn(obs, act, done)

        # sample next action
        next_act, logp = policy.sample_action(next_obs)

        # compute value target
        q1_next, q2_next = critic_target(next_obs, next_act)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - beta * logp
        q_target = compute_q_target(r, v_next, done, gamma, use_terminal=use_terminal)
    
    q1, q2 = critic(obs, act)
    q1_loss = torch.pow(q1 - q_target, 2).mean()
    q2_loss = torch.pow(q2 - q_target, 2).mean()
    q_loss = (q1_loss + q2_loss) / 2 
    return q_loss