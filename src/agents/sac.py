import time
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
import torch.distributions.transforms as torch_transform

# model imports
from src.agents.nn_models import MLP, DoubleQNetwork
from src.agents.rl_utils import ReplayBuffer, Logger

class TanhTransform(torch_transform.Transform):
    """ Adapted from Pytorch implementation with clipping """
    domain = torch_dist.constraints.real
    codomain = torch_dist.constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, limits):
        super().__init__() 
        assert isinstance(limits, torch.Tensor)
        self.limits = nn.Parameter(limits, requires_grad=False)
        self.eps = 1e-5

    def __call__(self, x):
        return self.limits * torch.tanh(x)
    
    def _inverse(self, y):
        y = torch.clip(y / self.limits, -1. + self.eps, 1. - self.eps) # prevent overflow
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        ldj = (2. * (np.log(2.) - x - F.softplus(-2. * x)))
        ldj += torch.abs(self.limits).log()
        return ldj


class SAC(nn.Module):
    """ Soft actor critic """
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        act_lim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        polyak=0.995, 
        tune_beta=True, 
        buffer_size=int(1e6), 
        batch_size=100, 
        steps=50, 
        lr_a=1e-3, 
        lr_c=1e-3, 
        grad_clip=None
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            act_lim (torch.tensor): action limits
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            steps (int, optional): actor critic update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-3
            lr_c (float, optional): critic learning rate. Default=1e-3
            grad_clip (float, optional): gradient clipping. Default=None
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_lim = act_lim
        self.gamma = gamma
        self.beta = beta
        self.polyak = polyak
        self.tune_beta = tune_beta
        self.beta_target = -act_dim # default temperature target
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.steps = steps
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.grad_clip = grad_clip
        
        self.log_beta = nn.Parameter(np.log(beta) * torch.ones(1), requires_grad=tune_beta)
        self.actor = MLP(obs_dim, act_dim * 2, hidden_dim, num_hidden, activation)
        self.critic = DoubleQNetwork(
            obs_dim, act_dim, hidden_dim, num_hidden, activation
        )
        self.critic_target = deepcopy(self.critic)

        # freeze target parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.optimizers = {
            "actor": torch.optim.Adam(
                self.actor.parameters(), lr=lr_a
            ),
            "critic": torch.optim.Adam(
                self.critic.parameters(), lr=lr_c
            ),
            "beta": torch.optim.Adam(
                [self.log_beta], lr=lr_a
            )
        }
        
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.99)
        
        self.plot_keys = ["eval_eps_return_avg", "eval_eps_len_avg", "critic_loss_avg", "actor_loss_avg", "beta_avg"]
    
    def sample_action(self, obs):
        mu, lv = torch.chunk(self.actor.forward(obs), 2, dim=-1)
        std = torch.exp(lv.clip(np.log(1e-3), np.log(100)))
        base_dist = torch_dist.Normal(mu, std)
        act = base_dist.rsample()
        logp = base_dist.log_prob(act).sum(-1, keepdim=True)

        ldj = (2. * (np.log(2.) - act - F.softplus(-2. * act)))
        ldj += torch.abs(self.act_lim).log()

        act = torch.tanh(act) * self.act_lim
        logp -= ldj.sum(-1, keepdim=True)
        return act, logp
    
    def compute_action_likelihood(self, obs, act):
        eps = 1e-5
        act_inv = torch.clip(act / self.act_lim, -1. + eps, 1. - eps) # prevent overflow
        act_inv = torch.atanh(act_inv)

        mu, lv = torch.chunk(self.actor.forward(obs), 2, dim=-1)
        std = torch.exp(lv.clip(np.log(1e-3), np.log(100)))
        base_dist = torch_dist.Normal(mu, std)
        return base_dist.log_prob(act_inv).sum(-1, keepdim=True)

    def choose_action(self, obs):
        with torch.no_grad():
            a, _ = self.sample_action(obs)
        return a

    def compute_critic_loss(self, batch, rwd_fn=None):
        obs = batch["obs"]
        act = batch["act"]
        r = batch["rwd"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        
        with torch.no_grad():
            if rwd_fn is not None:
                r = rwd_fn(obs, act)

            # sample next action
            next_act, logp = self.sample_action(next_obs)

            # compute value target
            q1_next, q2_next = self.critic_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next)
            v_next = q_next - self.beta * logp
            q_target = r + (1 - done) * self.gamma * v_next
        
        q1, q2 = self.critic(obs, act)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2 
        return q_loss
    
    def compute_actor_loss(self, batch):
        obs = batch["obs"]
        
        act, logp = self.sample_action(obs)
        
        q1, q2 = self.critic(obs, act)
        q = torch.min(q1, q2)

        a_loss = torch.mean(self.beta * logp - q)
        beta_loss = -torch.mean(self.log_beta * (logp + self.beta_target).detach())
        return a_loss, beta_loss

    def take_policy_gradient_step(self, batch, rwd_fn=None):
        self.actor.train()
        self.critic.train()
        
        # train critic
        critic_loss = self.compute_critic_loss(batch, rwd_fn)
        critic_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizers["critic"].step()
        self.optimizers["critic"].zero_grad()
        self.optimizers["actor"].zero_grad()

        # train actor
        actor_loss, beta_loss = self.compute_actor_loss(batch)
        actor_loss.backward()
        if self.tune_beta:
            beta_loss.backward()
        
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.optimizers["actor"].step()
        self.optimizers["beta"].step()
        self.optimizers["actor"].zero_grad()
        self.optimizers["beta"].zero_grad()
        self.optimizers["critic"].zero_grad()
        
        # update target networks and temperature
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
            
            self.beta = self.log_beta.exp().data

        stats = {
            "actor_loss": actor_loss.data.item(),
            "critic_loss": critic_loss.data.item(),
            "beta_loss": beta_loss.data.item(),
            "beta": self.beta,
        }
        
        self.actor.eval()
        self.critic.eval()
        return stats
    
    def rollout(self, env, max_steps):
        obs = env.reset()[0]

        data = {"obs": [], "act": [], "next_obs": [], "rwd": [], "done": []}
        for t in range(max_steps):
            with torch.no_grad():
                act = self.choose_action(
                    torch.from_numpy(obs).to(torch.float32)
                ).numpy()
            next_obs, rwd, terminated, _, _ = env.step(act)
            
            data["obs"].append(obs)
            data["act"].append(act)
            data["next_obs"].append(next_obs)
            data["rwd"].append(rwd)
            data["done"].append(terminated)

            if terminated:
                break
            
            obs = next_obs

        data["obs"] = torch.from_numpy(np.stack(data["obs"])).to(torch.float32)
        data["act"] = torch.from_numpy(np.stack(data["act"])).to(torch.float32)
        data["next_obs"] = torch.from_numpy(np.stack(data["next_obs"])).to(torch.float32)
        data["rwd"] = torch.from_numpy(np.stack(data["rwd"])).to(torch.float32)
        data["done"] = torch.from_numpy(np.stack(data["done"])).to(torch.float32)
        return data
    
    def train_policy_epoch(self, rwd_fn=None, logger=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            batch = self.replay_buffer.sample(self.batch_size)
            policy_stats = self.take_policy_gradient_step(batch, rwd_fn=rwd_fn)
            policy_stats_epoch.append(policy_stats)

            if logger is not None:
                logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch

    def train_policy(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, update_every, 
        rwd_fn=None, num_eval_eps=0, callback=None, verbose=True
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
        obs, eps_return, eps_len = env.reset()[0], 0, 0
        for t in range(total_steps):
            if (t + 1) < update_after:
                act = torch.rand(self.act_dim).uniform_(-1, 1) * self.act_lim
                act = act.data.numpy()
            else:
                with torch.no_grad():
                    act = self.choose_action(
                        torch.from_numpy(obs).view(1, -1).to(torch.float32)
                    ).numpy().flatten()
            next_obs, reward, terminated, truncated, info = env.step(act)
            
            eps_return += reward
            eps_len += 1
            
            self.replay_buffer.push(
                obs, act, reward, next_obs, np.array(1. * terminated)
            )
            obs = next_obs
            
            # end of trajectory handeling
            if terminated or (eps_len + 1) > max_steps:
                self.replay_buffer.push_batch()
                logger.push({"eps_return": eps_return})
                logger.push({"eps_len": eps_len})
                
                # start new episode
                obs, eps_return, eps_len = env.reset()[0], 0, 0

            # train model
            if (t + 1) > update_after and (t - update_after + 1) % update_every == 0:
                policy_stats_epoch = self.train_policy_epoch(rwd_fn=rwd_fn, logger=logger)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) > update_after and (t - update_after + 1) % steps_per_epoch == 0:
                epoch = (t - update_after + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    eval_eps = []
                    for i in range(num_eval_eps):
                        eval_eps.append(self.rollout(eval_env, max_steps))
                        logger.push({"eval_eps_return": sum(eval_eps[-1]["rwd"])})
                        logger.push({"eval_eps_len": sum(1 - eval_eps[-1]["done"])})

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()

                if t > update_after and callback is not None:
                    callback(self, logger)
        
        env.close()
        return logger
        