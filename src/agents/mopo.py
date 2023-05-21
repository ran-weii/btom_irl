import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.mbpo import MBPO
from src.utils.evaluation import evaluate_episodes
from src.utils.logger import Logger

class MOPO(MBPO):
    """ Model-based offline policy optimization 
    
    With automatic penalty tuning based on https://arxiv.org/abs/2110.04135
    """
    def __init__(
        self, 
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
        lam=1.,
        lam_target=1.5,
        tune_lam=True,
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
        a_steps=1, 
        lr_a=3e-4, 
        lr_c=3e-4, 
        lr_lam=0.01, 
        grad_clip=None,
        device=torch.device("cpu")
        ):
        """
        Args:
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
            a_steps (int, optional): policy update steps per training step. Default=1
            lr_a (float, optional): actor learning rate. Default=3e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_lam (float, optional): penalty learning rate. Default=0.01
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            dynamics, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, min_beta, polyak, tune_beta, False, buffer_size, batch_size, 
            rollout_batch_size, rollout_deterministic, rollout_min_steps, rollout_max_steps, 
            rollout_min_epoch, rollout_max_epoch, model_retain_epochs, None,
            real_ratio, eval_ratio, a_steps, a_steps, lr_a, lr_c, lr_a, grad_clip, device
        )
        self.lam = lam
        self.lam_target = lam_target
        self.tune_lam = tune_lam

        self.log_lam = nn.Parameter(np.log(lam) * torch.ones(1), requires_grad=tune_beta)
        self.optimizers["lambda"] = torch.optim.Adam(
            [self.log_lam], lr=lr_lam
        )

        self.plot_keys = [
            "eval_eps_return_mean", "eval_eps_len_mean", "critic_loss", 
            "actor_loss", "beta", "lam", 
        ]
    
    def compute_penalty(self, obs, act):
        """ Compute ensemble standard deviation penalty """
        dist = self.dynamics.compute_dists(obs, act)
        mean = dist.mean
        variance = dist.variance
        mean_of_vars = torch.mean(variance, dim=-2)
        var_of_means = torch.var(mean, dim=-2)
        std = (mean_of_vars + var_of_means).sqrt()
        pen = torch.mean(std, dim=-1, keepdim=True)
        return pen
    
    def update_lambda(self, mopo_pen):
        lam = self.log_lam.exp()
        lam_loss = torch.mean(self.log_lam * (lam * mopo_pen - self.lam_target).detach())
        
        if self.tune_lam:
            lam_loss.backward()
            self.optimizers["lambda"].step()
            self.optimizers["lambda"].zero_grad()

        with torch.no_grad():
            self.lam = self.log_lam.exp().data
    
    def rollout_dynamics(
        self, obs, act=None, rollout_steps=5, 
        rollout_deterministic=False, terminate_early=True, flatten=True
        ):
        """ Rollout dynamics model from initial states with mopo penalty

        Args:
            obs (torch.tensor): initial observations. size=[batch_size, obs_dim]
            act (torch.tensor, optional): initial actions. size=[batch_size, act_dim]
            rollout_steps (int, optional): number of rollout steps. Default=5
            rollout_deterministic (bool, optional): whether to rollout deterministically. Default=False
            terminate_early (bool, optional): whether to terminate before rollout steps. Default=True
            flatten (bool, optional): whether to flatten trajectories into transitions. Default=True

        Returns:
            data (dict): dictionary of trajectories or transitions. size=[rollout_steps, batch_size, dim]
        """
        self.dynamics.eval()
        
        obs = obs.clone()
        data = {"obs": [], "act": [], "next_obs": [], "rwd": [], "done": []}
        for t in range(rollout_steps):
            with torch.no_grad():
                if t == 0 and act is not None:
                    pass
                else:
                    act = self.choose_action(obs)
                next_obs, rwd, done = self.dynamics.step(obs, act, sample_mean=rollout_deterministic)
                mopo_pen = self.compute_penalty(obs, act)
            
            self.update_lambda(mopo_pen)

            data["obs"].append(obs)
            data["act"].append(act)
            data["next_obs"].append(next_obs)
            data["rwd"].append(rwd - self.lam * mopo_pen)
            data["done"].append(done)
            
            if terminate_early:
                obs = next_obs[done.flatten() == 0].clone()
            else:
                obs = next_obs.clone()
            
            if len(obs) == 0:
                break

        if flatten:
            data = {k: torch.cat(v, dim=0) for k, v in data.items()}
        else:
            data = {k: torch.stack(v) for k, v in data.items()}
        return data
    
    def train_policy(
        self, eval_env, epochs, steps_per_epoch, sample_model_every, 
        num_eval_eps=10, eval_steps=1000, eval_deterministic=True, callback=None, verbose=10
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
                    evaluate_episodes(eval_env, self, num_eval_eps, eval_steps, eval_deterministic, logger)

                logger.push({"epoch": epoch})
                logger.push({"time": time.time() - start_time})
                logger.push({"lam": self.lam.cpu().item()})
                logger.log()
                print()

                if callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        return logger
