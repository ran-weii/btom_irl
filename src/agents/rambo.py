import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.mbpo import MBPO
from src.utils.data import normalize, denormalize
from src.utils.logging import Logger

class RAMBO(MBPO):
    """ Robust adversarial model-based offline policy optimization """
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
        gamma=0.9, 
        beta=0.2, 
        polyak=0.995, 
        tune_beta=True,
        obs_penalty=1., 
        adv_penalty=3e-4, 
        adv_clip_max=6.,
        norm_advantage=False,
        update_critic_adv=False,
        buffer_size=1e6, 
        batch_size=200, 
        rollout_batch_size=10000, 
        rollout_min_steps=1, 
        rollout_max_steps=10, 
        rollout_min_epoch=20, 
        rollout_max_epoch=100, 
        model_retain_epochs=5,
        real_ratio=0.05, 
        eval_ratio=0.2,
        m_steps=100, 
        a_steps=50, 
        lr_a=1e-4, 
        lr_c=3e-4, 
        lr_m=3e-4, 
        grad_clip=None,
        device=torch.device("cpu")
        ):
        """
        Args:
            reward (EnsembleDynamics): reward function as an EnsembleDynamics object
            dynamics (EnsembleDynamics): transition function as an EnsembleDynamics object
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
            obs_penalty (float, optional): transition likelihood penalty. Default=1..
            adv_penalty (float, optional): model advantage penalty. Default=3e-4.
            adv_clip_max (float, optional): advantage clipping threshold. Default=6.
            norm_advantage (bool, optional): whether to normalize advantage. Default=False
            update_adv_critic (bool, optional): whether to udpate critic during model update. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            rollout_batch_size (int, optional): model_rollout batch size. Default=10000
            rollout_min_steps (int, optional): initial model rollout steps. Default=1
            rollout_max_steps (int, optional): maximum model rollout steps. Default=10
            rollout_min_epoch (int, optional): epoch to start increasing rollout length. Default=20
            rollout_max_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            model_retain_epochs (int, optional): number of epochs to keep model samples. Default=5
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.05
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            m_steps (int, optional): model update steps per training step. Default=100
            a_steps (int, optional): policy update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_m (float, optional): model learning rate. Default=3e-4
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            reward, dynamics, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, tune_beta, False, buffer_size, batch_size, 
            rollout_batch_size, rollout_min_steps, rollout_max_steps, 
            rollout_min_epoch, rollout_max_epoch, model_retain_epochs,
            real_ratio, eval_ratio, m_steps, a_steps, lr_a, lr_c, lr_m, grad_clip, device
        )
        self.obs_penalty = obs_penalty
        self.adv_penalty = adv_penalty
        self.adv_clip_max = adv_clip_max
        self.norm_advantage = norm_advantage
        self.update_critic_adv = update_critic_adv
        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "critic_loss_avg", 
            "actor_loss_avg", "beta_avg", "adv_loss_avg", "obs_mae_avg", 
        ]
    
    def compute_dynamics_adversarial_loss(self, obs, act):
        # sample next obs and rwd
        with torch.no_grad():
            rwd, _ = self.reward.step(obs, act)
            next_obs, done = self.dynamics.step(obs, act)
            rwd_norm = normalize(rwd, self.reward.out_mean, self.reward.out_variance)
            obs_norm = normalize(obs, self.dynamics.out_mean, self.dynamics.out_variance)
            next_obs_norm = normalize(next_obs, self.dynamics.out_mean, self.dynamics.out_variance)
        
        # compute advantage
        q_1, q_2 = self.critic(obs, act)
        q = torch.min(q_1, q_2)
        with torch.no_grad():
            next_act, logp = self.sample_action(next_obs)
            q_next_1, q_next_2 = self.critic(next_obs, next_act)
            q_next = torch.min(q_next_1, q_next_2)
            v_next = q_next - self.beta * logp
            advantage = rwd + (1 - done) * self.gamma * v_next - q.data
            
            if self.norm_advantage:
                advantage_norm = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-6)
            else:
                advantage_norm = advantage
            advantage_norm = advantage_norm.clip(-self.adv_clip_max, self.adv_clip_max)
        
        # compute ensemble mixture log likelihood
        logp_rwd = self.reward.compute_mixture_log_prob(obs_norm, act, rwd_norm)
        logp_obs = self.dynamics.compute_mixture_log_prob(obs_norm, act, next_obs_norm)
        adv_loss = torch.mean(advantage_norm * (logp_rwd + logp_obs))

        # update model aware q loss
        with torch.no_grad():
            q_target_next_1, q_target_next_2 = self.critic_target(next_obs, next_act)
            q_target_next = torch.min(q_target_next_1, q_target_next_2)
            v_target_next = q_target_next - self.beta * logp
            q_target = rwd + (1 - done) * self.gamma * v_target_next

        q1_loss = torch.pow(q_1 - q_target, 2).mean()
        q2_loss = torch.pow(q_2 - q_target, 2).mean()
        adv_q_loss = (q1_loss + q2_loss) / 2 

        stats = {
            "v_next_mean": v_next.cpu().mean().data.item(),
            "adv_mean": advantage.cpu().mean().data.item(),
            "adv_std": advantage.cpu().std().data.item(),
            "logp_rwd_mean": logp_rwd.cpu().mean().data.item(),
            "logp_rwd_std": logp_rwd.cpu().std().data.item(),
            "logp_obs_mean": logp_obs.cpu().mean().data.item(),
            "logp_obs_std": logp_obs.cpu().std().data.item(),
        }
        return adv_loss, adv_q_loss, next_obs, stats
    
    def take_adversarial_model_gradient_step(self, obs, act, sl_batch):
        self.reward.train()
        self.dynamics.train()
        
        adv_loss, adv_q_loss, next_obs, adv_stats = self.compute_dynamics_adversarial_loss(obs, act)
        reward_loss = self.reward.compute_loss(sl_batch["obs"], sl_batch["act"], sl_batch["rwd"])
        dynamics_loss = self.dynamics.compute_loss(sl_batch["obs"], sl_batch["act"], sl_batch["next_obs"])
        total_loss = self.obs_penalty * (reward_loss + dynamics_loss) + self.adv_penalty * adv_loss
        total_loss.backward()
        if self.update_critic_adv:
            adv_q_loss.backward()
        
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizers["reward"].step()
        self.optimizers["dynamics"].step()
        if self.update_critic_adv:
            self.optimizers["critic"].step()
        self.optimizers["reward"].zero_grad()
        self.optimizers["dynamics"].zero_grad()
        self.optimizers["critic"].zero_grad()
        self.optimizers["actor"].zero_grad()
        
        # update target networks
        if self.update_critic_adv:
            with torch.no_grad():
                for p, p_target in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_target.data.mul_(self.polyak)
                    p_target.data.add_((1 - self.polyak) * p.data)

        stats = {
            "rwd_loss": reward_loss.cpu().data.item(),
            "obs_loss": dynamics_loss.cpu().data.item(),
            "adv_loss": adv_loss.cpu().data.item(),
            "critic_adv_loss": adv_q_loss.cpu().data.item(),
            **adv_stats
        }
        
        self.reward.eval()
        self.dynamics.eval()
        return stats, next_obs
    
    def train_adversarial_model_epoch(self, steps, rollout_steps, logger=None):
        # train test split supervised learning data
        num_total = min(int(self.batch_size * steps * (1 + self.eval_ratio)), self.real_buffer.size)
        num_eval = int(self.eval_ratio * num_total)
        data = self.real_buffer.sample(num_total)

        # normalize data
        data["obs"] = normalize(data["obs"], self.dynamics.obs_mean.cpu(), self.dynamics.obs_variance.cpu())
        data["next_obs"] = normalize(data["next_obs"], self.dynamics.obs_mean.cpu(), self.dynamics.obs_variance.cpu())
        data["rwd"] = normalize(data["rwd"], self.reward.out_mean.cpu(), self.reward.out_variance.cpu())

        train_data = {k:v[:-num_eval].to(self.device) for k, v in data.items()}
        eval_data = {k:v[-num_eval:].to(self.device) for k, v in data.items()}

        # shuffle train data
        idx_train = np.arange(len(train_data["obs"]))
        np.random.shuffle(idx_train)

        train_stats_epoch = []
        counter = 0
        idx_start_sl = 0
        while counter < steps:
            batch = self.real_buffer.sample(self.batch_size)
            obs = batch["obs"].to(self.device)
            for t in range(rollout_steps):
                # get supervised batch
                idx_batch = idx_train[idx_start_sl:idx_start_sl+self.batch_size]
                train_batch = {k:v[idx_batch] for k, v in train_data.items()}
                
                act = self.choose_action(obs)
                model_train_stats, next_obs = self.take_adversarial_model_gradient_step(
                    obs, act, train_batch
                )
                train_stats_epoch.append(model_train_stats)
                obs = next_obs.clone()
                
                if logger is not None:
                    logger.push(model_train_stats)

                counter += 1
                if counter == steps:
                    break

                # update sl batch counter
                idx_start_sl += self.batch_size
                if idx_start_sl + self.batch_size >= len(train_data["obs"]):
                    idx_start_sl = 0
                    idx_train = np.arange(len(train_data["obs"]))
                    np.random.shuffle(idx_train)

        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()

        # evaluate
        reward_eval_stats = self.reward.evaluate(eval_data["obs"], eval_data["act"], eval_data["rwd"])
        dynamics_eval_stats = self.dynamics.evaluate(eval_data["obs"], eval_data["act"], eval_data["next_obs"])
        self.reward.update_topk_dist(reward_eval_stats)
        self.dynamics.update_topk_dist(dynamics_eval_stats)
        eval_stats = {
            "rwd_mae": reward_eval_stats["mae"],
            "obs_mae": dynamics_eval_stats["mae"],
        }
        if logger is not None:
            logger.push(eval_stats)
        
        stats_epoch = {**train_stats_epoch, **eval_stats}
        return stats_epoch
    
    def train_policy(
        self, eval_env, max_steps, epochs, steps_per_epoch, sample_model_every, update_model_every,
        rwd_fn=None, num_eval_eps=0, eval_deterministic=True, callback=None, verbose=10
        ):
        logger = Logger()
        start_time = time.time()
        total_steps = epochs * steps_per_epoch
        
        epoch = 0
        for t in range(total_steps):
            # train model adversary
            if (t + 1) % update_model_every == 0:
                model_stats_epoch = self.train_adversarial_model_epoch(self.m_steps, rollout_steps, logger=logger)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in model_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")

            # sample model
            if t == 0 or (t + 1) % sample_model_every == 0:
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.replay_buffer.max_size = min(
                    self.buffer_size, int(self.model_retain_epochs * self.rollout_batch_size * rollout_steps)
                )
                self.sample_imagined_data(
                    self.rollout_batch_size, rollout_steps, mix=False
                )
                print("rollout_steps: {}, real buffer size: {}, fake buffer size: {}".format(
                    rollout_steps, self.real_buffer.size, self.replay_buffer.size
                ))

            # train policy
            policy_stats_epoch = self.train_policy_epoch(
                logger=logger
            )
            if (t + 1) % verbose == 0:
                round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0: 
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    eval_eps = []
                    eval_returns_est = []
                    eval_returns = []
                    eval_lens = []
                    for i in range(num_eval_eps):
                        eval_eps.append(self.rollout(eval_env, max_steps, sample_mean=eval_deterministic))

                        # compute estimated return 
                        with torch.no_grad():
                            r, _ = self.reward.step(
                                eval_eps[-1]["obs"].to(self.device),
                                eval_eps[-1]["act"].to(self.device)
                            )

                        eval_returns_est.append(r.cpu().sum().data.item())
                        eval_returns.append(sum(eval_eps[-1]["rwd"]))
                        eval_lens.append(sum(1 - eval_eps[-1]["done"]))

                        logger.push({"eval_eps_return_est": r.cpu().sum().data.item()})
                        logger.push({"eval_eps_return": sum(eval_eps[-1]["rwd"])})
                        logger.push({"eval_eps_len": sum(1 - eval_eps[-1]["done"])})

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()

                if callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        return logger
