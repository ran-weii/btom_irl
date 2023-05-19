import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from src.agents.nn_models import MLP

class Reward(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        state_only=False,
        clip_max=10.,
        decay=0.,
        grad_penalty=1.,
        grad_target=0.,
        device=torch.device("cpu")
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            state_only (bool, optional): whether to use state-only reward. Default=False
            clip_max (float, optional): clip reward max value. Default=10.
            d_decay (float, optional): reward weight decay. Default=0.
            grad_penalty (float, optional): reward gradient penalty weight. Default=1.
            grad_target (float, optional): reward gradient penalty target. Default=1.
            device (optional): training device. Default=cpu
        """
        super().__init__()
        self.state_only = state_only
        self.clip_max = clip_max
        self.decay = decay
        self.grad_penalty = grad_penalty
        self.grad_target = grad_target
        self.device = device

        self.mlp = MLP(
            obs_dim + act_dim * (not state_only) + 1, 
            1, 
            hidden_dim, 
            num_hidden, 
            activation
        )

    def forward(self, obs, act, done):
        """ Compute clipped and done masked reward """
        obs_mask = obs * (1 - done)
        act_mask = act * (1 - done)
        if self.state_only:
            rwd_inputs = torch.cat([obs_mask, done], dim=-1)
        else:
            rwd_inputs = torch.cat([obs_mask, act_mask, done], dim=-1)
        return self.mlp.forward(rwd_inputs).clip(-self.clip_max, self.clip_max)

    def compute_loss_traj(self, real_batch, fake_batch, gamma):
        real_obs = real_batch["obs"].to(self.device)
        real_act = real_batch["act"].to(self.device)
        real_done = real_batch["done"].to(self.device)

        fake_obs = fake_batch["obs"].to(self.device)
        fake_act = fake_batch["act"].to(self.device)
        fake_done = fake_batch["done"].to(self.device)

        real_rwd = self.forward(real_obs, real_act, real_done)
        fake_rwd = self.forward(fake_obs, fake_act, fake_done)
        
        gamma_seq = gamma ** torch.arange(len(real_rwd)).view(-1, 1, 1)
        real_return = torch.sum(gamma_seq * real_rwd, dim=0)
        fake_return = torch.sum(gamma_seq * fake_rwd, dim=0)

        d_loss = -(real_return.mean() - fake_return.mean())
        return d_loss

    def compute_loss_marginal(self, real_batch, fake_batch):
        real_obs = real_batch["obs"].to(self.device)
        real_act = real_batch["act"].to(self.device)
        real_done = real_batch["done"].to(self.device)

        fake_obs = fake_batch["obs"].to(self.device)
        fake_act = fake_batch["act"].to(self.device)
        fake_done = fake_batch["done"].to(self.device)

        real_rwd = self.forward(real_obs, real_act, real_done)
        fake_rwd = self.forward(fake_obs, fake_act, fake_done)

        d_loss = -(real_rwd.mean() - fake_rwd.mean())
        return d_loss

    def compute_grad_penalty(self, real_batch, fake_batch):
        """ Compute two-sided gradient penalty """
        real_obs = real_batch["obs"].to(self.device)
        real_act = real_batch["act"].to(self.device)
        real_done = real_batch["done"].to(self.device)

        fake_obs = fake_batch["obs"].to(self.device)
        fake_act = fake_batch["act"].to(self.device)
        fake_done = fake_batch["done"].to(self.device)
        
        obs_dim = real_obs.shape[-1]
        act_dim = real_act.shape[-1]
        
        real_inputs = torch.cat([real_obs, real_act, real_done], dim=-1)
        fake_inputs = torch.cat([fake_obs, fake_act, real_done], dim=-1)
        
        alpha = torch.rand(len(real_inputs), 1)
        interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs

        interpolated = Variable(interpolated, requires_grad=True)
        obs_var, act_var, done_var = torch.split(interpolated, [obs_dim, act_dim, 1], dim=-1)

        rwd = self.forward(obs_var, act_var, done_var)
        
        grad = torch_grad(
            outputs=rwd, inputs=interpolated, 
            grad_outputs=torch.ones_like(rwd),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen 
    
    def compute_decay_loss(self):
        loss = 0
        for layer in self.mlp.layers:
            if hasattr(layer, "weight"):
                loss += torch.sum(layer.weight ** 2) / 2.
                loss += torch.sum(layer.bias ** 2) / 2.
        return loss