import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.mbpo import MBPO
from src.agents.rl_utils import EpisodeReplayBuffer, Logger

class OfflineIRL(MBPO):
    """ Offline model-based inverse reinforcement learning """
    def __init__(
        self, obs_dim, act_dim, act_lim, 
        ensemble_dim, hidden_dim, num_hidden, activation, 
        gamma=0.9, beta=0.2, polyak=0.995, clip_lv=False, 
        rollout_steps=10, buffer_size=1000000, d_batch_size=10, a_batch_size=200,
        rollout_batch_size=10000, d_steps=3, a_steps=50, lr_d=3e-4, lr_a=0.001, 
        decay=0, grad_clip=None, pess_penalty=0.
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
            m_steps (int, optional): model update steps per training step. Default=100
            a_steps (int, optional): policy update steps per training step. Default=50
            lr (float, optional): learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0.
            grad_clip (float, optional): gradient clipping. Default=None
        """
        super().__init__(
            obs_dim, act_dim, act_lim, ensemble_dim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, clip_lv, rollout_steps, buffer_size, a_batch_size, 
            rollout_batch_size, None, None, None, 
            None, a_steps, lr_a, decay, grad_clip
        )
        assert d_steps > 1
        self.pess_penalty = pess_penalty
        self.d_batch_size = d_batch_size
        self.a_batch_size = a_batch_size
        self.d_steps = d_steps
        self.grad_clip = grad_clip
        
        self.optimizers["reward"].param_groups[0]["lr"] = lr_d
        
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
    
    def compute_reward_with_penalty(self, obs, act):
        r = self.compute_reward(obs, act)
        
        with torch.no_grad():
            ensemble_std = self.compute_transition_dist(
                obs, act
            ).sample().std(-2).sum(-1, keepdim=True)
        return r - self.pess_penalty * ensemble_std

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

    # def take_reward_gradient_step(self, fake_batch, fake_mask, logger=None):
    def take_reward_gradient_step(self, max_steps, logger=None):
        self.reward.train()

        reward_loss_epoch = []
        for i in range(self.d_steps):
            # collect fake samples
            real_batch, _ = self.real_buffer.sample_episodes(self.d_batch_size)
            real_obs = real_batch["obs"][0]
            real_done = real_batch["done"][0]
            fake_batch = self.rollout_dynamics(real_obs, real_done, max_steps)
            fake_mask = torch.ones(max_steps, self.d_batch_size)
            
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
    
    def train_model_epoch(self, train_loader, eval_loader, logger):
        model_train_stats_epoch = []
        for i, train_batch in enumerate(train_loader):
            model_train_stats = self.take_model_gradient_step(train_batch)
            model_train_stats_epoch.append(model_train_stats)
            logger.push(model_train_stats)
        model_train_stats_epoch = pd.DataFrame(model_train_stats_epoch).mean(0).to_dict()
        
        model_eval_stats_epoch = []
        for i, eval_batch in enumerate(eval_loader):
            model_eval_stats = self.eval_model(eval_batch)
            model_eval_stats_epoch.append(model_eval_stats)
            logger.push(model_eval_stats)
        model_eval_stats_epoch = pd.DataFrame(model_eval_stats_epoch).mean(0).to_dict()

        model_stats_epoch = {**model_train_stats_epoch, **model_eval_stats_epoch}
        return model_stats_epoch

    def train_model_offline(self, train_loader, eval_loader, epochs, callback=None):
        logger = Logger()
        start_time = time.time()

        for e in range(epochs):
            self.train_model_epoch(train_loader, eval_loader, logger)

            # log stats
            logger.push({"epoch": e + 1})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

            if callback is not None:
                callback(self, logger)
        
        return logger
    
    def train_policy_epoch(self, logger, rwd_fn=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            # mix real and fake data
            # real_batch = self.real_buffer.sample(self.batch_size)
            # fake_batch = self.replay_buffer.sample(int(self.real_ratio * self.batch_size))
            # batch = {
            #     real_k: torch.cat([real_v, fake_v], dim=0) 
            #     for ((real_k, real_v), (fake_k, fake_v)) 
            #     in zip(real_batch.items(), fake_batch.items())
            # }

            batch = self.replay_buffer.sample(self.batch_size)
            policy_stats = self.take_policy_gradient_step(batch, rwd_fn=rwd_fn)
            policy_stats_epoch.append(policy_stats)
            logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch
    
    def train_policy(
        self, eval_env, max_steps, epochs, steps_per_epoch, update_after, 
        update_policy_every,
        rwd_fn=None, num_eval_eps=0, callback=None, verbose=True
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
        for t in range(total_steps):
            if t < update_after:
                batch = self.real_buffer.sample(self.rollout_batch_size)
            else:
                # real_batch = self.real_buffer.sample(int(self.rollout_batch_size/2))
                # fake_batch = self.replay_buffer.sample(int(self.rollout_batch_size/2))
                # batch = {
                #     real_k: torch.cat([real_v, fake_v], dim=0) 
                #     for ((real_k, real_v), (fake_k, fake_v)) 
                #     in zip(real_batch.items(), fake_batch.items())
                # }
                batch = self.real_buffer.sample(self.rollout_batch_size)

            rollout_data = self.rollout_dynamics(batch["obs"], batch["done"], self.rollout_steps)
            self.replay_buffer.push_batch(
                rollout_data["obs"].flatten(0, 1).numpy(),
                rollout_data["act"].flatten(0, 1).numpy(),
                rollout_data["rwd"].flatten(0, 1).numpy(),
                rollout_data["next_obs"].flatten(0, 1).numpy(),
                rollout_data["done"].flatten(0, 1).numpy()
            )
            
            # train policy
            if (t + 1) > update_after and (t - update_after + 1) % update_policy_every == 0:
                policy_stats_epoch = self.train_policy_epoch(logger, rwd_fn=rwd_fn)
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
        
        return logger

    def train(
        self, eval_env, max_steps, epochs, rl_epochs, steps_per_epoch, update_after, update_every,
        callback=None, verbose=True
        ):
        self.reward.eval()
        
        logger = Logger()
        start_time = time.time()
        for e in range(epochs):
            update_after_ = update_after if e == 0 else 0
            policy_logger = self.train_policy(
                eval_env, max_steps, rl_epochs, steps_per_epoch, update_after_, update_every, 
                rwd_fn=self.compute_reward_with_penalty, num_eval_eps=0, verbose=verbose
            )
            
            # # collect fake samples
            # real_batch, _ = self.real_buffer.sample_episodes(self.d_batch_size)
            # real_obs = real_batch["obs"][0]
            # real_done = real_batch["done"][0]
            # fake_batch = self.rollout_dynamics(real_obs, real_done, max_steps)
            # fake_mask = torch.ones(max_steps, self.d_batch_size)
            
            # self.take_reward_gradient_step(fake_batch, fake_mask, logger)

            self.take_reward_gradient_step(max_steps, logger)
            
            # evaluate
            eval_eps = []
            for i in range(5):
                eval_eps.append(self.rollout(eval_env, max_steps))
                logger.push({"eval_eps_return": sum(eval_eps[-1]["rwd"])})
                logger.push({"eval_eps_len": sum(1 - eval_eps[-1]["done"])})

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
