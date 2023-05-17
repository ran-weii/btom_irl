import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.rambo import RAMBO
from src.agents.buffer import ReplayBuffer, EpisodeReplayBuffer
from src.agents.critic import compute_q_target, compute_critic_loss
from src.utils.evaluation import evaluate_episodes, evaluate_policy
from src.utils.logger import Logger

class RTOM(RAMBO):
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
        beta=1., 
        min_beta=0.001, 
        polyak=0.995, 
        tune_beta=True, 
        rwd_rollout_batch_size=64,
        rwd_rollout_steps=100,
        rwd_update_method="traj",
        obs_penalty=1, 
        adv_penalty=0.1, 
        adv_rollout_steps=10, 
        adv_action_deterministic=True, 
        adv_include_entropy=False, 
        adv_clip_max=6., 
        norm_advantage=True, 
        buffer_size=2e6, 
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
        d_steps=1,
        m_steps=50, 
        a_steps=1, 
        lr_a=0.0003, 
        lr_c=0.0003, 
        lr_d=0.0001,
        lr_m=0.0001, 
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
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            min_beta (float, optional): minimum softmax temperature. Default=0.001
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            rwd_rollout_batch_size (int, optional): reward rollout batch size. Default=64
            rwd_rollout_steps (int, optional): reward rollout steps. Default=100
            rwd_update_method (str, optional): reward update method. Choices=["traj", "marginal"]
            obs_penalty (float, optional): transition likelihood penalty. Default=1.
            adv_penalty (float, optional): model advantage penalty. Default=0.1
            adv_rollout_steps (int, optional): adversarial rollout steps. Default=10.
            adv_action_deterministic (bool, optional): whether to use deterministic action in advantage. Default=True
            adv_include_entropy (bool, optional): whether to include entropy in advantage. Default=False
            adv_clip_max (float, optional): advantage clipping threshold. Default=6.
            norm_advantage (bool, optional): whether to normalize advantage. Default=True
            buffer_size (int, optional): replay buffer size. Default=2e6
            batch_size (int, optional): actor and critic batch size. Default=256
            rollout_batch_size (int, optional): model_rollout batch size. Default=50000
            rollout_deterministic (bool, optional): whether to rollout deterministically. Default=False
            rollout_min_steps (int, optional): initial model rollout steps. Default=5
            rollout_max_steps (int, optional): maximum model rollout steps. Default=5
            rollout_min_epoch (int, optional): epoch to start increasing rollout length. Default=20
            rollout_max_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            model_retain_epochs (int, optional): number of epochs to keep model samples. Default=5
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.5
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            d_steps (int, optional): reward update steps per training step. Default=1
            m_steps (int, optional): model update steps per training step. Default=50
            a_steps (int, optional): policy update steps per training step. Default=1
            lr_a (float, optional): actor learning rate. Default=3e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_d (float, optional): reward learning rate. Default=1e-4
            lr_m (float, optional): model learning rate. Default=1e-4
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            dynamics, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, min_beta, polyak, tune_beta, obs_penalty, adv_penalty, 
            adv_rollout_steps, adv_action_deterministic, adv_include_entropy, 
            adv_clip_max, norm_advantage, buffer_size, batch_size, 
            rollout_batch_size, rollout_deterministic, rollout_min_steps, 
            rollout_max_steps, rollout_min_epoch, rollout_max_epoch, 
            model_retain_epochs, real_ratio, eval_ratio, 
            m_steps, a_steps, lr_a, lr_c, lr_m, grad_clip, device
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
            "eval_eps_return_mean", "eval_eps_len_mean", "rwd_loss", "adv_loss", 
            "mae", "log_pi", "critic_loss", "actor_loss",  "beta"
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
    
    def compute_dynamics_adversarial_loss(self, obs, act):
        # sample next obs and rwd
        with torch.no_grad():
            next_obs, _, done = self.dynamics.step(obs, act)
        
        # compute advantage
        with torch.no_grad():
            q_1, q_2 = self.critic(obs, act)
            q = torch.min(q_1, q_2)

            rwd = self.reward.forward(obs, act, done)
            
            next_act, logp = self.sample_action(next_obs, sample_mean=self.adv_action_deterministic)
            q_next_1, q_next_2 = self.critic(next_obs, next_act)
            q_next = torch.min(q_next_1, q_next_2)
            v_next = q_next - self.adv_include_entropy * (self.beta * logp)
            advantage = compute_q_target(rwd, v_next, done, self.gamma, use_terminal=True) - q.data
            
            if self.norm_advantage:
                advantage_norm = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-6)
            else:
                advantage_norm = advantage
            advantage_norm = advantage_norm.clip(-self.adv_clip_max, self.adv_clip_max)
        
        # compute ensemble mixture log likelihood
        logp = self.dynamics.compute_mixture_log_prob(obs, act, next_obs, None)
        adv_loss = torch.mean(advantage_norm * logp) / self.obs_dim * self.dynamics.topk
        
        # compute critic loss stats
        with torch.no_grad():
            q_target_next_1, q_target_next_2 = self.critic_target(next_obs, next_act)
            q_target_next = torch.min(q_target_next_1, q_target_next_2)
            v_target_next = q_target_next - self.beta * logp
            q_target = compute_q_target(rwd, v_target_next, done, self.gamma, use_terminal=True)

            q1_loss = torch.pow(q_1 - q_target, 2).mean()
            q2_loss = torch.pow(q_2 - q_target, 2).mean()
            adv_q_loss = (q1_loss + q2_loss) / 2 

        stats = {
            "adv_v_next_mean": v_next.cpu().mean().data.item(),
            "adv_mean": advantage.cpu().mean().data.item(),
            "adv_std": advantage.cpu().std().data.item(),
            "adv_logp_mean": logp.cpu().mean().data.item(),
            "adv_logp_std": logp.cpu().std().data.item(),
            "adv_critic_loss": adv_q_loss.cpu().data.item(),
        }
        return adv_loss, next_obs, stats
    
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
    
    def compute_critic_loss(self, batch):
        return compute_critic_loss(
            batch, self, self.critic, self.critic_target,
            self.gamma, self.beta, self.device, 
            rwd_fn=self.reward.forward, use_terminal=True
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

            # train model adversary
            if (t + 1) % update_model_every == 0:
                reward_stats_epoch = self.train_reward_epoch(logger=logger)
                dynamics_stats_epoch = self.train_adversarial_model_epoch(self.m_steps, logger=logger)
                model_stats_epoch = {**reward_stats_epoch, **dynamics_stats_epoch}
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in model_stats_epoch.items()}
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
                logger.log()
                print()

                if callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        return logger
