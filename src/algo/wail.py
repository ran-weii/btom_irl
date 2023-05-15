import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# model imports
from src.agents.sac import SAC
from src.agents.buffer import EpisodeReplayBuffer
from src.agents.critic import compute_critic_loss
from src.utils.evaluate import evaluate
from src.utils.logger import Logger

class WAIL(SAC):
    """ Wasserstein adversarial imitation learning """
    def __init__(
        self, 
        reward,
        obs_dim, 
        act_dim, 
        act_lim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        min_beta=0.2,
        polyak=0.995, 
        tune_beta=False, 
        buffer_size=1e6, 
        batch_size=100, 
        real_ratio=0.5,
        d_steps=50, 
        a_steps=50, 
        lr_a=0.001, 
        lr_c=0.001, 
        lr_d=0.001,
        grad_clip=None, 
        device=torch.device("cpu")
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
            min_beta (float, optional): minimum softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            real_ratio (float, optional): policy training batch real ratio. Default=0.5
            d_steps (int, optional): reward update steps per training step. Default=50
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-3
            lr_c (float, optional): critic learning rate. Default=1e-3
            lr_d (float, optional): reward learning rate. Default=1e-3
            grad_clip (float, optional): gradient clipping. Default=None
            device (optional): training device. Default=cpu
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, min_beta, polyak, tune_beta, buffer_size, batch_size, 
            a_steps, lr_a, lr_c, grad_clip, device
        )
        self.real_ratio = real_ratio
        self.d_steps = d_steps

        self.reward = reward
        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr_d, weight_decay=self.reward.decay
        )

        self.expert_buffer = EpisodeReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.)
        
        self.plot_keys = [
            "eval_eps_return_mean", "eval_eps_len_mean", "rwd_loss", "log_pi",
            "critic_loss", "actor_loss", "beta"
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

    def train_reward_epoch(self, logger=None):
        reward_stats_epoch = []
        for _ in range(self.d_steps):
            real_batch = self.expert_buffer.sample(int(self.batch_size/2))
            fake_batch = self.replay_buffer.sample(int(self.batch_size/2))
            
            rwd_loss = self.reward.compute_loss_marginal(real_batch, fake_batch)
            gp = self.reward.compute_grad_penalty(real_batch, fake_batch)
            reward_total_loss = rwd_loss + self.reward.grad_penalty * gp
            
            reward_total_loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.reward.parameters(), self.grad_clip)
            self.optimizers["reward"].step()
            self.optimizers["reward"].zero_grad()

            reward_stats = {
                "rwd_loss": rwd_loss.data.item(),
                "grad_pen": gp.data.item(),
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

    def train_policy_epoch(self, logger=None):
        policy_stats_epoch = []
        for _ in range(self.steps):
            # mix real and fake data
            real_batch = self.expert_buffer.sample(int(self.real_ratio * self.batch_size))
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

    def train(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, update_every, 
        num_eval_eps=10, eval_steps=1000, eval_deterministic=True, callback=None, verbose=50
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
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
            
            self.replay_buffer.push(
                obs, act, np.array([0.]), next_obs, np.array(1. * terminated)
            )
            obs = next_obs
            
            # end of trajectory handeling
            if terminated or (eps_len + 1) > max_steps:
                self.replay_buffer.push_batch()
                logger.push({"eps_return": eps_return})
                logger.push({"eps_len": eps_len})
                
                # start new episode
                obs, eps_return, eps_len = env.reset()[0], 0, 0

            # train model
            if (t + 1) > update_after and (t - update_after + 1) % update_every == 0:
                reward_stats_epoch = self.train_reward_epoch(logger=logger)
                policy_stats_epoch = self.train_policy_epoch(logger=logger)
                stats_epoch = {**reward_stats_epoch, **policy_stats_epoch}
                if (t + 1) % verbose == 0:
                    round_loss_dict = {k: round(v, 3) for k, v in stats_epoch.items()}
                    print(f"e: {epoch + 1}, t: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) > update_after and (t - update_after + 1) % steps_per_epoch == 0:
                epoch = (t - update_after + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    evaluate(eval_env, self, num_eval_eps, eval_steps, eval_deterministic, logger)
                
                # evaluate policy
                with torch.no_grad():
                    batch = self.expert_buffer.sample(1000)
                    log_pi = self.compute_action_likelihood(batch["obs"],batch["act"])
                    logger.push({"log_pi": log_pi.cpu().mean().data.item()})

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()

                if t > update_after and callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        env.close()
        return logger
        