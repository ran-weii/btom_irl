import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# model imports
from src.agents.sac import SAC
from src.agents.nn_models import MLP
from src.agents.buffer import EpisodeReplayBuffer
from src.utils.logging import Logger

class WAIL(SAC):
    """ Wasserstein adversarial imitation learning """
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        act_lim, 
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        polyak=0.995, 
        tune_beta=False, 
        rwd_clip_max=10.,
        buffer_size=1e6, 
        batch_size=100, 
        real_ratio=0.5,
        d_steps=50, 
        a_steps=50, 
        lr_a=0.001, 
        lr_c=0.001, 
        lr_d=0.001,
        decay=0.,
        grad_clip=None, 
        grad_penalty=1., 
        grad_target=1.,
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
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=False
            rwd_clip_max (float, optional): clip reward max value. Default=10.
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): actor and critic batch size. Default=100
            real_ratio (float, optional): policy training batch real ratio. Default=0.5
            d_steps (int, optional): reward update steps per training step. Default=50
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr_a (float, optional): actor learning rate. Default=1e-3
            lr_c (float, optional): critic learning rate. Default=1e-3
            lr_d (float, optional): reward learning rate. Default=1e-3
            decay (float, optional): reward weight decay. Default=0.
            grad_clip (float, optional): gradient clipping. Default=None
            grad_penalty (float, optional): gradient penalty weight. Default=1.
            grad_target (float, optional): gradient penalty target. Default1.
            device (optional): training device. Default=cpu
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, tune_beta, buffer_size, batch_size, a_steps, 
            lr_a, lr_c, grad_clip, device
        )
        self.rwd_clip_max = rwd_clip_max
        self.real_ratio = real_ratio
        self.d_steps = d_steps
        self.grad_penalty = grad_penalty
        self.grad_target = grad_target

        self.reward = MLP(
            obs_dim + act_dim + 1, 1, hidden_dim, num_hidden, activation
        )

        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr_d, weight_decay=decay
        )

        self.real_buffer = EpisodeReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.)
        
        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "reward_loss_avg", 
            "critic_loss_avg", "actor_loss_avg", "log_pi_avg"
        ]
    
    def fill_real_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = batch["obs"]
            act = batch["act"]
            next_obs = batch["next_obs"]
            rwd = np.zeros((len(obs), 1))
            done = batch["done"].reshape(-1, 1)
            self.real_buffer.push(obs, act, rwd, next_obs, done)

    def compute_reward(self, obs, act, done):
        return self.reward(torch.cat([obs, act, done], dim=-1)).clip(-self.rwd_clip_max, self.rwd_clip_max)

    def compute_reward_loss(self, real_batch, fake_batch):
        real_done = real_batch["done"].to(self.device)
        real_obs = real_batch["obs"].to(self.device) * (1 - real_done)
        real_act = real_batch["act"].to(self.device) * (1 - real_done)
        
        fake_done = fake_batch["done"].to(self.device)
        fake_obs = fake_batch["obs"].to(self.device) * (1 - fake_done)
        fake_act = fake_batch["act"].to(self.device) * (1 - fake_done)

        real_rwd = self.compute_reward(real_obs, real_act, real_done)
        fake_rwd = self.compute_reward(fake_obs, fake_act, fake_done)

        d_loss = -(real_rwd.mean() - fake_rwd.mean())
        return d_loss

    def compute_grad_penalty(self, real_batch):
        """ Score matching gradient penalty """
        real_done = real_batch["done"].to(self.device)
        real_obs = real_batch["obs"].to(self.device) * (1 - real_done)
        real_act = real_batch["act"].to(self.device) * (1 - real_done)
        
        real_inputs = torch.cat([real_obs, real_act, real_done], dim=-1)
        real_var = Variable(real_inputs, requires_grad=True)
        obs_var, act_var, done_var = torch.split(real_var, [self.obs_dim, self.act_dim, 1], dim=-1)

        rwd = self.compute_reward(obs_var, act_var, done_var)
        
        grad = torch_grad(
            outputs=rwd, inputs=real_var, 
            grad_outputs=torch.ones_like(rwd),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen 

    def train_reward_epoch(self, logger=None):
        reward_stats_epoch = []
        for _ in range(self.d_steps):
            real_batch = self.real_buffer.sample(int(self.batch_size/2))
            fake_batch = self.replay_buffer.sample(int(self.batch_size/2))
            
            reward_loss = self.compute_reward_loss(real_batch, fake_batch)
            gp = self.compute_grad_penalty(real_batch)
            reward_total_loss = reward_loss + self.grad_penalty * gp
            
            reward_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.reward.parameters(), self.grad_clip)
            self.optimizers["reward"].step()
            self.optimizers["reward"].zero_grad()

            reward_stats = {
                "reward_loss": reward_loss.data.item(),
                "grad_pen": gp.data.item(),
            }
            reward_stats_epoch.append(reward_stats)
            if logger is not None:
                logger.push(reward_stats)
    
        reward_stats_epoch = pd.DataFrame(reward_stats_epoch).mean(0).to_dict()
        return reward_stats_epoch
    
    def compute_critic_loss(self, batch, rwd_fn=None):
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            if rwd_fn is not None:
                r = rwd_fn(obs, act, done)
                r_done = rwd_fn(
                    torch.zeros(1, self.obs_dim).to(torch.float32).to(self.device),
                    torch.zeros(1, self.act_dim).to(torch.float32).to(self.device),
                    torch.ones(1, 1).to(torch.float32).to(self.device)
                ) # special handle terminal state

            # sample next action
            next_act, logp = self.sample_action(next_obs)

            # compute value target
            q1_next, q2_next = self.critic_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next)
            v_next = q_next - self.beta * logp
            v_done = self.gamma / (1 - self.gamma) * r_done # special handle terminal state
            q_target = r + (1 - done) * self.gamma * v_next + done * v_done
        
        q1, q2 = self.critic(obs, act)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2 
        return q_loss

    def train_policy_epoch(self, rwd_fn=None, logger=None):
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
            policy_stats = self.take_policy_gradient_step(batch, rwd_fn=rwd_fn)

            # eval policy
            with torch.no_grad():
                log_pi = self.compute_action_likelihood(
                    real_batch["obs"].to(self.device),
                    real_batch["act"].to(self.device)
                )

            policy_stats["log_pi"] = log_pi.mean().cpu().data.item()
            policy_stats_epoch.append(policy_stats)
            if logger is not None:
                logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch

    def train(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, update_every, 
        eval_steps=1000, num_eval_eps=0, eval_deterministic=True, callback=None, verbose=50
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
                policy_stats_epoch = self.train_policy_epoch(
                    rwd_fn=self.compute_reward, logger=logger
                )
                stats_epoch = {**reward_stats_epoch, **policy_stats_epoch}
                if (t + 1) % verbose == 0:
                    round_loss_dict = {k: round(v, 3) for k, v in stats_epoch.items()}
                    print(f"e: {epoch + 1}, t: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) > update_after and (t - update_after + 1) % steps_per_epoch == 0:
                epoch = (t - update_after + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    eval_eps = []
                    for i in range(num_eval_eps):
                        eval_eps.append(self.rollout(eval_env, eval_steps, sample_mean=eval_deterministic))
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
        