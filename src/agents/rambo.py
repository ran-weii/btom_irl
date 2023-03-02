import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.mbpo import MBPO
from src.agents.rl_utils import Logger
from src.agents.rl_utils import normalize, denormalize

class RAMBO(MBPO):
    """ Robust adversarial model-based offline policy optimization """
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        act_lim, 
        ensemble_dim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        polyak=0.995, 
        tune_beta=True,
        clip_lv=False, 
        rwd_clip_max=10., 
        obs_penalty=10., 
        buffer_size=1e6, 
        batch_size=200, 
        rollout_steps=10, 
        rollout_batch_size=10000, 
        topk=5,
        rollout_min_epoch=20, 
        rollout_max_epoch=100, 
        termination_fn=None, 
        real_ratio=0.05, 
        eval_ratio=0.2,
        m_steps=100, 
        a_steps=50, 
        lr_a=1e-4, 
        lr_c=3e-4, 
        lr_m=3e-4, 
        decay=None, 
        grad_clip=None,
        device=torch.device("cpu")
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
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            clip_lv (bool, optional): whether to soft clip observation log variance. Default=False
            rwd_clip_max (float, optional): clip reward max value. Default=10.
            obs_penalty (float, optional): observation likelihood penalty. Default=10.
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            rollout_batch_size (int, optional): model_rollout batch size. Default=10000
            rollout_steps (int, optional): number of model rollout steps. Default=10
            topk (int, optional): top k models to perform rollout. Default=5
            min_rollout_epoch (int, optional): epoch to start increasing rollout length. Default=20
            max_rollout_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            termination_fn (func, optional): termination function to output rollout done. Default=None
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.05
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            m_steps (int, optional): model update steps per training step. Default=100
            a_steps (int, optional): policy update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_m (float, optional): model learning rate. Default=3e-4
            decay ([list, None], optional): weight decay for each dynamics and reward model layer. Default=None.
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            obs_dim, act_dim, act_lim, ensemble_dim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, tune_beta, clip_lv, rwd_clip_max, False, buffer_size, batch_size, 
            rollout_batch_size, rollout_steps, topk, rollout_min_epoch, rollout_max_epoch, 
            termination_fn, real_ratio, eval_ratio, m_steps, a_steps, lr_a, lr_c, lr_m, decay, grad_clip, device
        )
        self.obs_penalty = obs_penalty
        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "critic_loss_avg", 
            "actor_loss_avg", "beta_avg", "adv_loss_avg", "obs_mae", 
        ]
    
    def compute_dynamics_adversarial_loss(self, batch):
        obs_norm = batch["obs"].to(self.device)
        done = batch["done"].to(self.device)
        
        # sample act and next obs
        obs = denormalize(obs_norm, self.obs_mean, self.obs_variance)
        with torch.no_grad():
            act, _ = self.sample_action(obs)
            rwd_norm = self.sample_reward_dist(obs_norm, act)
            next_obs_norm = self.sample_transition_dist(obs_norm, act)
            rwd = denormalize(rwd_norm, self.rwd_mean, self.rwd_variance)
            next_obs = denormalize(next_obs_norm, self.obs_mean, self.obs_variance)

        # compute model advantage
        if self.termination_fn is not None:
            done = self.termination_fn(
                obs.cpu().data.numpy(), act.cpu().data.numpy(), next_obs.cpu().data.numpy()
            )
            done = torch.from_numpy(done).view(-1, 1).to(torch.float32).to(self.device)
        
        # compute advantage
        with torch.no_grad():
            q_1, q_2 = self.critic(obs, act)
            q = torch.min(q_1, q_2)
            
            next_act, logp = self.sample_action(next_obs)
            q_next_1, q_next_2 = self.critic(next_obs, next_act)
            q_next = torch.min(q_next_1, q_next_2)
            v_next = q_next - self.beta * logp

            advantage = rwd + (1 - done) * self.gamma * v_next - q

            # normalize advantage
            advantage = (advantage - advantage.mean(0)) / advantage.std(0)
        
        logp_rwd = self.compute_reward_log_prob(obs_norm, act, rwd_norm).sum(-1)
        logp_obs = self.compute_transition_log_prob(obs_norm, act, next_obs_norm).sum(-1)
        loss = torch.mean(advantage * (logp_rwd + logp_obs))
        return loss
    
    def take_adversarial_model_gradient_step(self, real_batch):
        self.reward.train()
        self.dynamics.train()
        
        adversarial_loss = self.compute_dynamics_adversarial_loss(real_batch)
        reward_loss = self.compute_reward_loss(real_batch)
        dynamics_loss = self.compute_dynamics_loss(real_batch)
        total_loss = reward_loss + dynamics_loss + self.obs_penalty * adversarial_loss
        total_loss.backward()

        self.optimizers["dynamics"].step()
        self.optimizers["dynamics"].zero_grad()
        self.optimizers["actor"].zero_grad()
        self.optimizers["critic"].zero_grad()
        
        stats = {
            "rwd_loss": reward_loss.cpu().data.item(),
            "obs_loss": dynamics_loss.cpu().data.item(),
            "adv_loss": adversarial_loss.cpu().data.item()
        }
        
        self.reward.eval()
        self.dynamics.eval()
        return stats

    def train_adversarial_model_epoch(self, steps, logger=None, verbose=False):
        # train test split
        num_total = min(self.batch_size * steps, self.real_buffer.size)
        num_eval = int(self.eval_ratio * num_total)
        data = self.real_buffer.sample(num_total)

        # normalize data
        data["obs"] = normalize(data["obs"], self.obs_mean.cpu(), self.obs_variance.cpu())
        data["next_obs"] = normalize(data["next_obs"], self.obs_mean.cpu(), self.obs_variance.cpu())
        data["rwd"] = normalize(data["rwd"], self.rwd_mean.cpu(), self.rwd_variance.cpu())
        
        train_data = {k:v[:-num_eval] for k, v in data.items()}
        eval_data = {k:v[-num_eval:] for k, v in data.items()}

        # shuffle train data
        idx_train = np.arange(len(train_data["obs"]))
        np.random.shuffle(idx_train)
        
        train_stats_epoch = []
        for i in range(0, train_data["obs"].shape[0], self.batch_size):
            idx_batch = idx_train[i:i+self.batch_size]
            train_batch = {k:v[idx_batch] for k, v in train_data.items()}
            # fake_batch = self.replay_buffer.sample(self.batch_size)

            model_train_stats = self.take_adversarial_model_gradient_step(train_batch)
            train_stats_epoch.append(model_train_stats)
            
            # log stats
            if logger is not None:
                logger.push(model_train_stats)
        
        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()

        # evaluate
        reward_eval_stats = self.eval_reward(eval_data)
        model_eval_stats = self.eval_model(eval_data)
        logger.push({**reward_eval_stats, **model_eval_stats})

        stats_epoch = {**train_stats_epoch, **reward_eval_stats, **model_eval_stats}
        return stats_epoch

    def train_policy(
        self, eval_env, max_steps, epochs, steps_per_epoch, sample_model_every, 
        rwd_fn=None, num_eval_eps=0, eval_deterministic=True, callback=None, verbose=True
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        self.update_stats()
        self.sample_imagined_data(
            self.rollout_batch_size, self.rollout_steps, mix=False
        )
        
        epoch = 0
        for t in range(total_steps):
            # train model
            if (t + 1) % sample_model_every == 0:
                # generate imagined data
                self.replay_buffer.clear()
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.sample_imagined_data(
                    self.rollout_batch_size, rollout_steps, mix=False
                )
                print("replay buffer size", self.replay_buffer.size, "rollout steps", rollout_steps)

            # train policy
            policy_stats_epoch = self.train_policy_epoch(
                logger=logger
            )
            if (t + 1) % verbose == 0:
                round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

            # train model adversary
            if (t + 1) % steps_per_epoch == 0:
                model_stats_epoch = self.train_adversarial_model_epoch(self.m_steps, logger=logger)
                self.compute_topk_dist(model_stats_epoch)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in model_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0: 
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    eval_eps = []
                    for i in range(num_eval_eps):
                        eval_eps.append(self.rollout(eval_env, max_steps, sample_mean=eval_deterministic))

                        # compute estimated return 
                        with torch.no_grad():
                            r_norm = self.sample_reward_dist(
                                eval_eps[-1]["obs"].to(self.device), eval_eps[-1]["act"].to(self.device)
                            )
                            r = denormalize(r_norm, self.rwd_mean, self.rwd_variance)
                        logger.push({"eval_eps_est_return": sum(r.cpu())})
                        logger.push({"eval_eps_return": sum(eval_eps[-1]["rwd"])})
                        logger.push({"eval_eps_len": sum(1 - eval_eps[-1]["done"])})

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()

                if callback is not None:
                    callback(self, logger)
        
        return logger
