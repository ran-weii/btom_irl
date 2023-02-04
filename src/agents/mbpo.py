import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

from src.agents.sac import SAC
from src.agents.nn_models import MLP, EnsembleMLP
from src.agents.rl_utils import ReplayBuffer, Logger

class MBPO(SAC):
    """ Model-based policy optimization """
    def __init__(
        self, obs_dim, act_dim, act_lim, ensemble_dim, hidden_dim, num_hidden, activation, 
        gamma=0.9, beta=0.2, polyak=0.995, clip_lv=False, rollout_steps=10, buffer_size=1e6, 
        batch_size=200, rollout_batch_size=10000, rollout_min_epoch=20, rollout_max_epoch=100, 
        real_ratio=0.05, m_steps=100, a_steps=50, lr=0.001, decay=0, grad_clip=None
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            act_lim (torch.tensor): action limits
            ensemble_dim (int): number of ensemble models
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            clip_lv (bool, optional): whether to soft clip observation log variance. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            rollout_batch_size (int, optional): model_rollout batch size. Default=10000
            min_rollout_epoch (int, optional): epoch to start increasing rollout length. Default=20
            max_rollout_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.05
            m_steps (int, optional): model update steps per training step. Default=100
            a_steps (int, optional): policy update steps per training step. Default=50
            lr (float, optional): learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0.
            grad_clip (float, optional): gradient clipping. Default=None
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, buffer_size, batch_size, a_steps, 
            lr, decay, grad_clip
        )
        self.ensemble_dim = ensemble_dim
        self.clip_lv = clip_lv
        
        self.rollout_batch_size = rollout_batch_size
        self.rollout_min_epoch = rollout_min_epoch # used to calculate rollout steps
        self.rollout_max_epoch = rollout_max_epoch # used to calculate rollout steps
        self.rollout_steps = rollout_steps
        self.real_ratio = real_ratio
        self.m_steps = m_steps

        self.reward = MLP(
            obs_dim + act_dim, 1, hidden_dim, num_hidden, activation
        )
        self.dynamics = EnsembleMLP(
            obs_dim + act_dim, obs_dim * 2, ensemble_dim, hidden_dim, num_hidden, activation
        )
        
        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr, weight_decay=decay
        )
        self.optimizers["dynamics"] = torch.optim.Adam(
            self.dynamics.parameters(), lr=lr, weight_decay=decay
        )
        
        # buffer to store environment data
        self.real_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.9)

        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "critic_loss_avg", 
            "actor_loss_avg", "rwd_mae_avg", "obs_mae_avg"
        ]

        self.max_model_lv = nn.Parameter(torch.ones(self.obs_dim) / 2, requires_grad=False)
        self.min_model_lv = nn.Parameter(-torch.ones(self.obs_dim) * 10, requires_grad=False)

    def compute_reward(self, obs, act):
        r = self.reward.forward(torch.cat([obs, act], dim=-1)).clip(-10, 10)
        return r
    
    def compute_transition_dist(self, obs, act):
        obs_act = torch.cat([obs, act], dim=-1)
        mu, lv = torch.chunk(self.dynamics.forward(obs_act), 2, dim=-1)

        if not self.clip_lv:
            std = torch.exp(lv.clip(np.log(1e-3), np.log(3)))
        else:
            lv = self.max_model_lv - F.softplus(self.max_model_lv - lv)
            lv = self.min_model_lv + F.softplus(lv - self.min_model_lv)
            std = torch.exp(lv)
        return torch_dist.Normal(mu, std)
    
    def compute_transition_log_prob(self, obs, act, next_obs):
        return self.compute_transition_dist(obs, act).log_prob(next_obs.unsqueeze(-2))
    
    def sample_transition_dist(self, obs, act):
        next_obs = self.compute_transition_dist(obs, act).rsample()
        
        # randomly select from the ensemble
        ensemble_idx = torch.randint(0, self.ensemble_dim, size=(len(obs),))
        next_obs = next_obs[torch.arange(len(obs)), ensemble_idx]
        return next_obs

    def compute_reward_loss(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        r = batch["rwd"]

        r_pred = self.compute_reward(obs, act)
        loss = torch.pow(r_pred - r, 2).mean()
        return loss

    def compute_dynamics_loss(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]

        logp = self.compute_transition_log_prob(obs, act, next_obs).sum(-1)
        loss = -logp.mean()
        return loss
    
    def eval_model(self, batch):
        self.reward.eval()
        self.dynamics.eval()

        obs = batch["obs"]
        act = batch["act"]
        rwd = batch["rwd"]
        next_obs = batch["next_obs"]

        with torch.no_grad():
            next_obs_pred = self.sample_transition_dist(obs, act)
            rwd_pred = self.compute_reward(obs, act)
            
        obs_mae = torch.abs(next_obs_pred - next_obs).mean()
        rwd_mae = torch.abs(rwd_pred - rwd).mean()
        
        stats = {
            "obs_mae": obs_mae.data.item(),
            "rwd_mae": rwd_mae.data.item()
        }
        return stats
    
    def take_model_gradient_step(self, batch):
        self.reward.train()
        self.dynamics.train()
        
        # train reward
        reward_loss = self.compute_reward_loss(batch)
        reward_loss.backward()
        self.optimizers["reward"].step()
        self.optimizers["reward"].zero_grad()

        # train dynamics
        dynamics_loss = self.compute_dynamics_loss(batch)
        dynamics_loss.backward()
        self.optimizers["dynamics"].step()
        self.optimizers["dynamics"].zero_grad()
        
        stats = {
            "rwd_loss": reward_loss.data.item(),
            "obs_loss": dynamics_loss.data.item(),
        }

        self.reward.eval()
        self.dynamics.eval()
        return stats

    def rollout_dynamics(self, real_batch, rollout_steps):
        self.reward.eval()
        self.dynamics.eval()

        obs = real_batch["obs"].clone()
        done = real_batch["done"].clone()
        for t in range(rollout_steps):
            with torch.no_grad():
                act = self.choose_action(obs)
                rwd = self.compute_reward(obs, act)
                next_obs = self.sample_transition_dist(obs, act)
            
            self.replay_buffer.push_batch(
                obs.numpy(), act.numpy(), rwd.numpy(), next_obs.numpy(), done.numpy()
            )
            
            obs = next_obs.clone()
    
    def train_model_epoch(self, logger):
        model_stats_epoch = []
        for _ in range(self.m_steps):
            train_batch = self.real_buffer.sample(self.batch_size)
            eval_batch = self.real_buffer.sample(int(self.batch_size * 0.3))
            model_train_stats = self.take_model_gradient_step(train_batch)
            model_eval_stats = self.eval_model(eval_batch)
            model_stats = {**model_train_stats, **model_eval_stats}
            model_stats_epoch.append(model_stats)
            logger.push(model_stats)

        model_stats_epoch = pd.DataFrame(model_stats_epoch).mean(0).to_dict()
        return model_stats_epoch

    def train_policy_epoch(self, logger, rwd_fn=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            # mix real and fake data
            real_batch = self.real_buffer.sample(self.batch_size)
            fake_batch = self.replay_buffer.sample(int(self.real_ratio * self.batch_size))
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
    
    def compute_rollout_steps(self, epoch):
        """ Linearly increate rollout steps based on epoch """
        min_rollout_steps = 1
        ratio = (epoch - self.rollout_min_epoch) / (self.rollout_max_epoch - self.rollout_min_epoch)
        rollout_steps = min(
            self.rollout_steps, max(
                min_rollout_steps, min_rollout_steps + ratio * (self.rollout_steps - min_rollout_steps)
            )
        )
        return int(rollout_steps)

    def train_policy(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, 
        update_model_every, update_policy_every,
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
            
            self.real_buffer.push(
                obs, act, reward, next_obs, np.array(1. * terminated)
            )
            obs = next_obs
            
            # end of trajectory handeling
            if terminated or (eps_len + 1) > max_steps:
                self.real_buffer.push_batch()
                logger.push({"eps_return": eps_return})
                logger.push({"eps_len": eps_len})
                
                # start new episode
                obs, eps_return, eps_len = env.reset()[0], 0, 0

            # train model
            if (t + 1) >= update_after and (t - update_after + 1) % update_model_every == 0:
                model_stats_epoch = self.train_model_epoch(logger)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in model_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")
                
                # generate imagined data
                self.replay_buffer.clear()
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                real_batch = self.real_buffer.sample(self.rollout_batch_size)
                self.rollout_dynamics(real_batch, rollout_steps)
                print("epoch", epoch, "rollout steps", rollout_steps)

            # train policy
            if (t + 1) > update_after and (t - update_after + 1) % update_policy_every == 0:
                policy_stats_epoch = self.train_policy_epoch(logger)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

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