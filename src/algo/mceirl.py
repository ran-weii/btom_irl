import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.sac import SAC
from src.agents.nn_models import MLP
from src.agents.rl_utils import EpisodeReplayBuffer, Logger
from src.agents.rl_utils import collate_fn

class MCEIRL(SAC):
    """ Maximum causal entropy inverse reinforcement learning with soft actor critic solver """
    def __init__(
        self, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
        gamma=0.9, beta=0.2, polyak=0.995, buffer_size=int(1e5), 
        d_batch_size=10, a_batch_size=200, d_steps=3, a_steps=50, 
        lr_d=3e-4, lr_a=1e-3, decay=0., grad_clip=None
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
            buffer_size (int, optional): replay buffer size. Default=1e5
            d_batch_size (int, optional): reward batch size. Default=10
            a_batch_size (int, optional): agent batch size. Default=200
            d_steps (int, optional): reward update steps per training step. Default=3
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr_d (float, optional): reward learning rate. Default=3e-4
            lr_a (float, optional): agent learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0.
            grad_clip (float, optional): gradient clipping. Default=None
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, buffer_size, a_batch_size, a_steps, 
            lr_a, decay, grad_clip, 
        )
        assert d_steps > 1
        self.gamma = gamma
        
        self.d_batch_size = d_batch_size
        self.a_batch_size = a_batch_size
        self.d_steps = d_steps
        self.grad_clip = grad_clip
        
        self.reward = MLP(obs_dim + act_dim, 1, hidden_dim, num_hidden, activation)

        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr_d, weight_decay=decay
        )

        self.real_buffer = EpisodeReplayBuffer(obs_dim, act_dim, buffer_size)
        
        self.plot_keys = ["eval_eps_return_avg", "eval_eps_len_avg", "reward_loss_avg", "critic_loss_avg", "actor_loss_avg"]

    def fill_real_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = batch["obs"]
            act = batch["act"]
            next_obs = batch["next_obs"]
            rwd = np.zeros((len(obs), 1))
            done = batch["done"].reshape(-1, 1)
            self.real_buffer.push(obs, act, rwd, next_obs, done)
    
    def compute_reward(self, obs, act):
        return self.reward(torch.cat([obs, act], dim=-1)).clip(-8, 8)
    
    def compute_reward_cumulents(self, obs, act, mask):
        r = self.compute_reward(obs, act).squeeze(-1)
        gamma = self.gamma ** torch.arange(obs.shape[0]).view(-1, 1)
        rho = torch.sum(gamma * r * mask, dim=0)
        return rho

    def compute_reward_loss(self, fake_batch, fake_mask):
        real_batch, real_mask = self.real_buffer.sample_episodes(self.d_batch_size, prioritize=False)
        
        real_obs = real_batch["obs"]
        real_act = real_batch["act"]

        fake_obs = fake_batch["obs"]
        fake_act = fake_batch["act"]

        r_cum_real = self.compute_reward_cumulents(real_obs, real_act, real_mask)
        r_cum_fake = self.compute_reward_cumulents(fake_obs, fake_act, fake_mask)
        r_loss = -(r_cum_real.mean() - r_cum_fake.mean())
        return r_loss

    def take_reward_gradient_step(self, fake_batch, logger=None):
        self.reward.train()
        
        # pad fake traj
        fake_batch, fake_mask = collate_fn(fake_batch)

        reward_loss_epoch = []
        for i in range(self.d_steps):
            # train reward
            reward_loss = self.compute_reward_loss(fake_batch, fake_mask)
            reward_total_loss = reward_loss 
            reward_total_loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.reward.parameters(), self.grad_clip)
            self.optimizers["reward"].step()
            self.optimizers["reward"].zero_grad()

            reward_loss_epoch.append(reward_loss.data.item())
            
            if logger is not None:
                logger.push({"reward_loss": reward_loss.data.item()})

        self.reward.eval()
        return
    
    def train_policy_epoch(self, logger, rwd_fn=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            # mix real and fake data
            real_batch = self.real_buffer.sample(self.batch_size)
            fake_batch = self.replay_buffer.sample(self.batch_size)
            batch = {
                real_k: torch.cat([real_v, fake_v], dim=0) 
                for ((real_k, real_v), (fake_k, fake_v)) 
                in zip(real_batch.items(), fake_batch.items())
            }

            policy_stats = self.take_policy_gradient_step(batch, rwd_fn=rwd_fn)
            policy_stats_epoch.append(policy_stats)
            logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch

    def train(
        self, env, eval_env, max_steps, epochs, rl_epochs, steps_per_epoch, update_after, update_every,
        callback=None, verbose=True
        ):
        self.reward.eval()
        
        logger = Logger()
        start_time = time.time()
        for e in range(epochs):
            update_after_ = update_after if e == 0 else 0
            policy_logger = self.train_policy(
                env, eval_env, max_steps, rl_epochs, steps_per_epoch, update_after_, update_every, 
                rwd_fn=self.compute_reward, num_eval_eps=0, verbose=verbose
            )
            
            # collect fake samples
            fake_batch = []
            for i in range(self.d_batch_size):
                fake_traj = self.rollout(eval_env, max_steps)
                fake_batch.append(fake_traj)
                logger.push({"eval_eps_return": fake_traj["rwd"].sum()})
                logger.push({"eval_eps_len": len(fake_traj["rwd"])})

            self.take_reward_gradient_step(fake_batch, logger)
            
            # log stats
            logger.push({"epoch": e + 1})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

            policy_stats = policy_logger.history[-1]
            policy_stats = {k: v for (k, v) in policy_stats.items() if "eps" not in k}
            policy_stats.pop("epoch")
            policy_stats.pop("time")
            logger.history[-1] = {**logger.history[-1], **policy_stats}
            
            if callback is not None:
                callback(self, logger)
        
        return logger


