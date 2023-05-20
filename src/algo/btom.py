import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.rambo import RAMBO
from src.agents.buffer import ReplayBuffer, EpisodeReplayBuffer
from src.agents.critic import compute_q_target, compute_critic_loss
from src.agents.dynamics import get_random_index, format_samples_for_training
from src.utils.evaluation import evaluate_episodes, evaluate_policy
from src.utils.logger import Logger

class BTOM(RAMBO):
    """ Bayesian theory of mind """
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
        min_beta=0.1, 
        polyak=0.995, 
        tune_beta=True, 
        rwd_rollout_batch_size=1000,
        rwd_rollout_steps=40,
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
            min_beta (float, optional): minimum softmax temperature. Default=0.1
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            rwd_rollout_batch_size (int, optional): reward rollout batch size. Default=1000
            rwd_rollout_steps (int, optional): reward rollout steps. Default=40
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
    
    def take_adversarial_model_gradient_step(self, obs_real, act_real, obs_fake, act_fake, sl_inputs, sl_targets):
        self.dynamics.train()
        
        adv_real_loss, next_obs_real, adv_real_stats = self.compute_dynamics_adversarial_loss(obs_real, act_real)
        adv_fake_loss, next_obs_fake, adv_fake_stats = self.compute_dynamics_adversarial_loss(obs_fake, act_fake)
        adv_loss = -(adv_real_loss - adv_fake_loss)
        sl_loss = self.dynamics.compute_loss(sl_inputs, sl_targets)
        total_loss = (
            self.adv_penalty * adv_loss + \
            self.obs_penalty * sl_loss
        )
        total_loss.backward()
        
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizers["dynamics"].step()
        self.optimizers["dynamics"].zero_grad()
        self.optimizers["critic"].zero_grad()
        self.optimizers["actor"].zero_grad()
        
        adv_real_stats = {k.replace("adv_", "adv_real_"): v for k, v in adv_real_stats.items()}
        adv_fake_stats = {k.replace("adv_", "adv_fake_"): v for k, v in adv_fake_stats.items()}
        stats = {
            "sl_loss": sl_loss.cpu().data.item(),
            "adv_loss": adv_loss.cpu().data.item(),
            **adv_real_stats, **adv_fake_stats
        }
        
        self.dynamics.eval()
        return stats, next_obs_real, next_obs_fake
    
    def train_adversarial_model_epoch(self, steps, logger=None):
        # train test split supervised learning data
        num_total = min(int(self.batch_size * steps * (1 + self.eval_ratio)), self.real_buffer.size)
        num_eval = int(self.eval_ratio * num_total)
        data = self.real_buffer.sample(num_total)

        inputs, targets = format_samples_for_training(
            data, 
            residual=self.dynamics.residual,
            pred_rwd=self.dynamics.pred_rwd
        )

        train_inputs = inputs[:-num_eval]
        train_targets = targets[:-num_eval]
        eval_inputs = inputs[-num_eval:]
        eval_targets = targets[-num_eval:]

        # shuffle train data
        ensemble_dim = self.dynamics.ensemble_dim 
        idx_train = get_random_index(len(train_inputs), ensemble_dim, bootstrap=True)

        train_stats_epoch = []
        counter = 0
        idx_start_sl = 0
        while counter < steps:
            batch = self.expert_buffer.sample(self.rwd_rollout_batch_size)
            obs_real = obs_fake = batch["obs"].to(self.device)
            act_real = batch["act"].to(self.device)
            act_fake = self.choose_action(obs_fake)
            for t in range(self.adv_rollout_steps):
                # get supervised batch
                idx_batch = idx_train[idx_start_sl:idx_start_sl+self.batch_size]
                inputs_batch = train_inputs[idx_batch]
                targets_batch = train_targets[idx_batch]
                
                if t > 0:
                    act_real = self.choose_action(obs_real)
                    act_fake = self.choose_action(obs_fake)
                model_train_stats, next_obs_real, next_obs_fake = self.take_adversarial_model_gradient_step(
                    obs_real, act_real, obs_fake, act_fake, inputs_batch, targets_batch
                )
                train_stats_epoch.append(model_train_stats)
                obs_real = next_obs_real.clone()
                obs_fake = next_obs_fake.clone()
                
                if logger is not None:
                    logger.push(model_train_stats)

                counter += 1
                if counter == steps:
                    break

                # update sl batch counter
                idx_start_sl += self.batch_size
                if idx_start_sl + self.batch_size >= len(train_inputs):
                    idx_start_sl = 0
                    idx_train = get_random_index(len(train_inputs), ensemble_dim)

        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()

        # evaluate
        eval_stats = self.dynamics.evaluate(eval_inputs, eval_targets)
        eval_stats = {"mae": eval_stats["mae"]}
        if logger is not None:
            logger.push(eval_stats)
        
        stats_epoch = {**train_stats_epoch, **eval_stats}
        return stats_epoch

    def train_reward_epoch(self, logger=None):
        reward_stats_epoch = []
        for _ in range(self.d_steps):
            batch = self.expert_buffer.sample(self.rwd_rollout_batch_size)
            real_traj = self.rollout_dynamics(
                batch["obs"].to(self.device), 
                act=batch["act"].to(self.device),
                rollout_steps=self.rwd_rollout_steps,
                rollout_deterministic=self.rollout_deterministic,
                terminate_early=False,
                flatten=False,
            )
            fake_traj = self.rollout_dynamics(
                batch["obs"].to(self.device), 
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
