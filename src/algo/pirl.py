import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.mopo import MOPO
from src.agents.buffer import ReplayBuffer, EpisodeReplayBuffer
from src.agents.critic import compute_critic_loss
from src.utils.evaluation import evaluate_episodes, evaluate_policy
from src.utils.logger import Logger

class PIRL(MOPO):
    """ Pessimistic inverse reinforcement learning 
    https://arxiv.org/abs/2302.07457
    """
    def __init__(
        self, 
        reward,
        dynamics, 
        obs_dim, 
        act_dim, 
        act_lim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.99, 
        beta=1, 
        min_beta=0.001, 
        polyak=0.995, 
        tune_beta=True, 
        rwd_rollout_batch_size=64,
        rwd_rollout_steps=100,
        rwd_update_method="traj",
        lam=1, 
        lam_target=1.5, 
        tune_lam=True, 
        buffer_size=2000000, 
        batch_size=256, 
        rollout_batch_size=50000, 
        rollout_deterministic=False, 
        rollout_min_steps=5, 
        rollout_max_steps=5, 
        rollout_min_epoch=20, 
        rollout_max_epoch=100, 
        model_retain_epochs=5, 
        real_ratio=0.5, 
        eval_ratio=0.2, 
        a_steps=1, 
        d_steps=1,
        lr_a=0.0003, 
        lr_c=0.0003, 
        lr_lam=0.01, 
        lr_d=0.0001,
        grad_clip=None, 
        device=torch.device("cpu")
        ):
        """
        Args:
            reward (Reward): reward function object
            dynamics (EnsembleDynamics): transition function as an EnsembleDynamics object
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            act_lim (torch.tensor): action limits
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            gamma (float, optional): discount factor. Default=0.99
            beta (float, optional): softmax temperature. Default=1.
            min_beta (float, optional): minimum softmax temperature. Default=0.001
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            rwd_rollout_batch_size (int, optional): reward rollout batch size. Default=64
            rwd_rollout_steps (int, optional): reward rollout steps. Default=100
            rwd_update_method (str, optional): reward update method. Choices=["traj", "marginal"]
            lam (float, optional): mopo penalty. Default=1.
            lam_target (float, optional): mopo penalty target. Default=1.5
            tune_lam (bool, optional): whether to automatically tune mopo penalty. Default=True
            buffer_size (int, optional): replay buffer size. Default=2e6
            batch_size (int, optional): actor and critic batch size. Default=256
            rollout_batch_size (int, optional): model_rollout batch size. Default=50000
            rollout_deterministic (bool, optional): whether to rollout deterministically. Default=False
            rollout_min_steps (int, optional): initial model rollout steps. Default=5
            rollout_max_steps (int, optional): maximum model rollout steps. Default=5
            rollout_min_epoch (int, optional): epoch to start increasing rollout length. Default=20
            rollout_max_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            model_retain_epochs (int, optional): number of epochs to keep model samples. Default=5
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.05
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            d_steps (int, optional): reward update steps per training step. Default=1
            a_steps (int, optional): policy update steps per training step. Default=1
            lr_a (float, optional): actor learning rate. Default=3e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_lam (float, optional): penalty learning rate. Default=0.01
            lr_d (float, optional): reward learning rate. Default=1e-4
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            dynamics, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, min_beta, polyak, tune_beta, lam, lam_target, tune_lam, 
            buffer_size, batch_size, rollout_batch_size, rollout_deterministic, 
            rollout_min_steps, rollout_max_steps, rollout_min_epoch, rollout_max_epoch, 
            model_retain_epochs, real_ratio, eval_ratio, a_steps, 
            lr_a, lr_c, lr_lam, grad_clip, device
        )
        self.rwd_rollout_batch_size = rwd_rollout_batch_size
        self.rwd_rollout_steps = rwd_rollout_steps
        self.rwd_update_method = rwd_update_method
        self.d_steps = d_steps

        self.reward = reward
        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr_d
        )
        
        self.expert_buffer = EpisodeReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.)
        
        rwd_rollout_buffer_size = self.rwd_rollout_batch_size * self.rwd_rollout_steps * self.model_retain_epochs
        self.reward_rollout_buffer = ReplayBuffer(obs_dim, act_dim, rwd_rollout_buffer_size, momentum=0.)
        self.plot_keys = [
            "eval_eps_return_mean", "eval_eps_len_mean", "rwd_loss", 
            "log_pi", "critic_loss", "actor_loss",  "beta", "lam"
        ]
    
    def fill_expert_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = batch["obs"]
            act = batch["act"]
            next_obs = batch["next_obs"]
            rwd = np.zeros((len(obs), 1))
            done = batch["done"].reshape(-1, 1)
            self.expert_buffer.push(obs, act, rwd, next_obs, done)

    def train_reward_epoch_traj(self, logger=None):
        reward_stats_epoch = []
        for _ in range(self.d_steps):
            real_traj, _ = self.expert_buffer.sample_episode_segments(
                self.rwd_rollout_batch_size, self.rwd_rollout_steps
            )
            fake_traj = self.rollout_dynamics(
                real_traj["obs"][0].to(self.device), 
                rollout_steps=self.rwd_rollout_steps,
                rollout_deterministic=self.rollout_deterministic,
                terminate_early=False,
                flatten=False,
            )
            
            rwd_loss = self.reward.compute_loss_traj(real_traj, fake_traj, self.gamma)
            l2_loss = self.reward.compute_decay_loss()
            reward_total_loss = rwd_loss + self.reward.decay * l2_loss
            
            reward_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.reward.parameters(), self.grad_clip)
            self.optimizers["reward"].step()
            self.optimizers["reward"].zero_grad()

            reward_stats = {
                "rwd_loss": rwd_loss.cpu().data.item() / self.rwd_rollout_steps,
                "l2_loss": l2_loss.cpu().data.item()
            }
            reward_stats_epoch.append(reward_stats)
            if logger is not None:
                logger.push(reward_stats)
    
        reward_stats_epoch = pd.DataFrame(reward_stats_epoch).mean(0).to_dict()
        return reward_stats_epoch
    
    def train_reward_epoch_marginal(self, logger=None):
        real_batch = self.expert_buffer.sample(int(self.rwd_rollout_batch_size))
        self.sample_imagined_data(
            self.expert_buffer, self.reward_rollout_buffer,
            self.rwd_rollout_batch_size, self.rwd_rollout_steps, self.rollout_deterministic
        )

        reward_stats_epoch = []
        for _ in range(self.d_steps):
            real_batch = self.expert_buffer.sample(int(self.batch_size/2))
            fake_batch = self.reward_rollout_buffer.sample(int(self.batch_size/2))
            
            rwd_loss = self.reward.compute_loss_marginal(real_batch, fake_batch)
            gp = self.reward.compute_grad_penalty(real_batch, fake_batch)
            reward_total_loss = rwd_loss + self.reward.grad_penalty * gp
            
            reward_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.reward.parameters(), self.grad_clip)
            self.optimizers["reward"].step()
            self.optimizers["reward"].zero_grad()

            reward_stats = {
                "rwd_loss": rwd_loss.cpu().data.item(),
                "grad_pen": gp.cpu().data.item(),
            }
            reward_stats_epoch.append(reward_stats)
            if logger is not None:
                logger.push(reward_stats)
    
        reward_stats_epoch = pd.DataFrame(reward_stats_epoch).mean(0).to_dict()
        return reward_stats_epoch
    
    def train_reward_epoch(self, logger=None):
        if self.rwd_update_method == "marginal":
            rwd_stats_epoch = self.train_reward_epoch_marginal(logger=logger)
        else:
            rwd_stats_epoch = self.train_reward_epoch_traj(logger=logger)
        return rwd_stats_epoch
    
    def compute_reward_with_penalty(self, obs, act, done):
        rwd = self.reward.forward(obs, act, done)
        pen = self.compute_penalty(obs, act)
        return rwd - self.lam * pen
    
    def compute_critic_loss(self, batch):
        return compute_critic_loss(
            batch, self, self.critic, self.critic_target,
            self.gamma, self.beta, self.device, 
            rwd_fn=self.compute_reward_with_penalty, use_terminal=True
        )

    def train(
        self, 
        eval_env, 
        epochs, 
        steps_per_epoch, 
        sample_model_every, 
        update_model_every,
        num_eval_eps=10, 
        eval_steps=10000,
        eval_deterministic=True, 
        callback=None, 
        verbose=50
        ):
        logger = Logger()
        start_time = time.time()
        total_steps = epochs * steps_per_epoch
        
        epoch = 0
        for t in range(total_steps):
            # sample model
            if t == 0 or (t + 1) % sample_model_every == 0:
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.replay_buffer.max_size = self.reallocate_buffer_size(
                    rollout_steps, steps_per_epoch, sample_model_every
                )
                self.sample_imagined_data(
                    self.real_buffer, self.replay_buffer, 
                    self.rollout_batch_size, rollout_steps, self.rollout_deterministic
                )
                print("rollout_steps: {}, real buffer size: {}, fake buffer size: {}".format(
                    rollout_steps, self.real_buffer.size, self.replay_buffer.size
                ))

            # train policy
            policy_stats_epoch = self.train_policy_epoch(logger=logger)
            if (t + 1) % verbose == 0:
                round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

            # train reward
            if (t + 1) % update_model_every == 0:
                reward_stats_epoch = self.train_reward_epoch(logger=logger)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in reward_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0: 
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    evaluate_episodes(eval_env, self, num_eval_eps, eval_steps, eval_deterministic, logger)

                # evaluate policy
                batch = self.expert_buffer.sample(1000)
                evaluate_policy(batch, self, logger)

                logger.push({"epoch": epoch})
                logger.push({"time": time.time() - start_time})
                logger.push({"lam": self.lam.cpu().item()})
                logger.log()
                print()

                if callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        return logger
