import time
import numpy as np
import pandas as pd
import torch

from src.agents.sac import SAC
from src.agents.dynamics import train_ensemble
from src.agents.buffer import ReplayBuffer
from src.utils.evaluation import evaluate_episodes
from src.utils.logger import Logger

def compute_linear_scale(min_val, max_val, min_step, max_step, t):
    """ Linearly increate value based on step """
    ratio = (t - min_step) / (max_step - min_step)
    ratio = max(0, min(ratio, 1))
    val = min_val + ratio * (max_val - min_val)
    return val

class MBPO(SAC):
    """ Model-based policy optimization """
    def __init__(
        self, 
        dynamics,
        obs_dim, 
        act_dim, 
        act_lim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        min_beta=0.001,
        polyak=0.995, 
        tune_beta=True, 
        norm_obs=True,
        buffer_size=1e6, 
        batch_size=200, 
        rollout_batch_size=10000, 
        rollout_deterministic=False,
        rollout_min_steps=1, 
        rollout_max_steps=10, 
        rollout_min_epoch=20, 
        rollout_max_epoch=100,
        model_retain_epochs=5,
        model_train_samples=100000,
        real_ratio=0.05, 
        eval_ratio=0.2, 
        m_steps=100, 
        a_steps=50, 
        lr_a=0.001, 
        lr_c=0.001, 
        lr_m=0.001, 
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
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            min_beta (float, optional): minimum softmax temperature. Default=0.001
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            norm_obs (bool, optional): whether to normalize observations. Default=True
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            rollout_batch_size (int, optional): model_rollout batch size. Default=10000
            rollout_deterministic (bool, optional): whether to rollout deterministically. Default=False
            rollout_min_steps (int, optional): initial model rollout steps. Default=1
            rollout_max_steps (int, optional): maximum model rollout steps. Default=10
            rollout_min_epoch (int, optional): epoch to start increasing rollout length. Default=20
            rollout_max_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            model_retain_epochs (int, optional): number of epochs to keep model samples. Default=5
            model_train_samples (int, optional): maximum number of samples for model training. Default=1e5
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.05
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            m_steps (int, optional): model update steps per training step. Default=100
            a_steps (int, optional): policy update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-3
            lr_c (float, optional): critic learning rate. Default=1e-3
            lr_m (float, optional): model learning rate. Default=1e-3
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, min_beta, polyak, tune_beta, buffer_size, batch_size, 
            a_steps, lr_a, lr_c, grad_clip, device
        )
        self.norm_obs = norm_obs
        self.rollout_batch_size = rollout_batch_size
        self.rollout_deterministic = rollout_deterministic
        self.rollout_min_steps = rollout_min_steps
        self.rollout_max_steps = rollout_max_steps
        self.rollout_min_epoch = rollout_min_epoch # used to calculate rollout steps
        self.rollout_max_epoch = rollout_max_epoch # used to calculate rollout steps
        self.model_retain_epochs = model_retain_epochs
        self.model_train_samples = model_train_samples
        self.real_ratio = real_ratio
        self.eval_ratio = eval_ratio
        self.m_steps = m_steps
        
        self.dynamics = dynamics
        
        self.optimizers["dynamics"] = torch.optim.Adam(
            self.dynamics.parameters(), lr=lr_m, 
        )
        
        # buffer to store environment data
        self.real_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.)

        self.plot_keys = [
            "eval_eps_return_mean", "eval_eps_len_mean", "critic_loss", 
            "actor_loss", "beta", "mae"
        ]
    
    def train_dynamics_epoch(self, steps, max_epochs_since_update=5, logger=None):
        data = self.real_buffer.sample(self.model_train_samples)
        
        train_logger = train_ensemble(
            data, 
            self, 
            self.optimizers["dynamics"],
            eval_ratio=self.eval_ratio, 
            batch_size=self.batch_size, 
            epochs=steps, 
            bootstrap=True,
            grad_clip=self.grad_clip, 
            update_elites=True,
            max_epoch_since_update=max_epochs_since_update,
        )
        stats = {
            "obs_loss": train_logger.history[-1]["loss"],
            "mae": train_logger.history[-1]["mae"],
        }
        if logger is not None:
            logger.push(stats)
        return stats
    
    def train_policy_epoch(self, logger=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            # mix real and fake data
            real_batch = self.real_buffer.sample(int(self.real_ratio * self.batch_size))
            fake_batch = self.replay_buffer.sample(int((1 - self.real_ratio) * self.batch_size))
            batch = {
                real_k: torch.cat([real_v, fake_v], dim=0) 
                for ((real_k, real_v), (fake_k, fake_v)) 
                in zip(real_batch.items(), fake_batch.items())
            }
            policy_stats = self.take_policy_gradient_step(batch)
            policy_stats_epoch.append(policy_stats)

            if logger is not None:
                logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch
    
    def rollout_dynamics(
        self, obs, act=None, rollout_steps=5, 
        rollout_deterministic=False, terminate_early=True, flatten=True
        ):
        """ Rollout dynamics model from initial states

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

            data["obs"].append(obs)
            data["act"].append(act)
            data["next_obs"].append(next_obs)
            data["rwd"].append(rwd)
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
    
    def sample_imagined_data(
        self, source_buffer, target_buffer, batch_size, rollout_steps, rollout_deterministic=False
        ):
        """ Sample model rollout data and add to replay buffer
        
        Args:
            source_buffer (ReplayBuffer): replay buffer to draw initial observations
            target_buffer (ReplayBuffer): replay buffer to store simulated samples
            batch_size (int): model rollout batch size
            rollout_steps (int): model rollout steps
            rollout_deterministic (bool, optional): whether to rollout deterministically. Default=False
        """
        batch = source_buffer.sample(batch_size)
        
        rollout_data = self.rollout_dynamics(
            batch["obs"].to(self.device), 
            rollout_steps=rollout_steps,
            rollout_deterministic=rollout_deterministic,
            terminate_early=True,
            flatten=True
        )
        target_buffer.push_batch(
            rollout_data["obs"].cpu().numpy(),
            rollout_data["act"].cpu().numpy(),
            rollout_data["rwd"].cpu().numpy(),
            rollout_data["next_obs"].cpu().numpy(),
            rollout_data["done"].cpu().numpy()
        )
    
    def compute_rollout_steps(self, epoch):
        """ Linearly increase rollout steps based on epoch """
        rollout_steps = compute_linear_scale(
            self.rollout_min_steps, 
            self.rollout_max_steps, 
            self.rollout_min_epoch, 
            self.rollout_max_epoch,
            epoch
        )
        return int(rollout_steps)
    
    def reallocate_buffer_size(self, rollout_steps, steps_per_epoch, sample_model_every):
        rollouts_per_epoch = self.rollout_batch_size * steps_per_epoch / sample_model_every
        model_steps_per_epoch = rollouts_per_epoch * rollout_steps
        buffer_size = min(
            self.buffer_size, int(self.model_retain_epochs * model_steps_per_epoch)
        )
        return buffer_size

    def train_policy(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, 
        update_model_every, update_policy_every, num_eval_eps=10,
        eval_steps=1000, eval_deterministic=True, callback=None, verbose=50
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
        ready_to_train = False
        obs, eps_return, eps_len = env.reset()[0], 0, 0
        for t in range(total_steps):
            if (t + 1) < update_after:
                act = torch.rand(self.act_dim).uniform_(-1, 1) * self.act_lim.cpu()
                act = act.data.numpy()
            else:
                with torch.no_grad():
                    act = self.choose_action(
                        torch.from_numpy(obs).view(1, -1).to(torch.float32).to(self.device)
                    ).cpu().numpy().flatten()
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

            # train policy
            if ready_to_train and (t - update_after + 1) % update_policy_every == 0:
                policy_stats_epoch = self.train_policy_epoch(logger=logger)
                if (t + 1) % verbose == 0:
                    round_loss_dict = {k: round(v, 3) for k, v in policy_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

            # train model
            if (t + 1) >= update_after and (t - update_after + 1) % update_model_every == 0:
                ready_to_train = (t + 1) >= update_after
                model_stats_epoch = self.train_dynamics_epoch(
                    self.m_steps, 
                    max_epochs_since_update=5, 
                    logger=logger
                )
                
                # generate imagined data
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.replay_buffer.max_size = self.reallocate_buffer_size(
                    rollout_steps, steps_per_epoch, update_model_every
                )
                self.sample_imagined_data(
                    self.real_buffer, self.replay_buffer, 
                    self.rollout_batch_size, rollout_steps, self.rollout_deterministic
                )
                print("rollout_steps: {}, real buffer size: {}, fake buffer size: {}".format(
                    rollout_steps, self.real_buffer.size, self.replay_buffer.size
                ))

            # end of epoch handeling
            if (t + 1) > update_after and (t - update_after + 1) % steps_per_epoch == 0:
                epoch = (t - update_after + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    evaluate_episodes(eval_env, self, num_eval_eps, eval_steps, eval_deterministic, logger)

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()

                if t > update_after and callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        env.close()
        return logger