import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

from src.agents.sac import SAC
from src.agents.nn_models import MLP, EnsembleMLP
from src.agents.rl_utils import ReplayBuffer, Logger
from src.agents.rl_utils import normalize, denormalize

class MBPO(SAC):
    """ Model-based policy optimization """
    def __init__(
        self, obs_dim, act_dim, act_lim, ensemble_dim, hidden_dim, num_hidden, activation, 
        gamma=0.9, beta=0.2, polyak=0.995, clip_lv=False, rwd_clip_max=10., buffer_size=1e6, batch_size=200, 
        rollout_batch_size=10000, rollout_steps=10, topk=5, rollout_min_epoch=20, rollout_max_epoch=100, 
        termination_fn=None, real_ratio=0.05, eval_ratio=0.2, m_steps=100, a_steps=50, lr=0.001, decay=None, grad_clip=None
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
            rwd_clip_max (float, optional): clip reward max value. Default=10.
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
            lr (float, optional): learning rate. Default=1e-3
            decay ([list, None], optional): weight decay for each dynamics and reward model layer. Default=None.
            grad_clip (float, optional): gradient clipping. Default=None
        """
        super().__init__(
            obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, buffer_size, batch_size, a_steps, 
            lr, grad_clip
        )
        self.ensemble_dim = ensemble_dim
        self.clip_lv = clip_lv
        self.rwd_clip_max = rwd_clip_max
        
        self.rollout_batch_size = rollout_batch_size
        self.rollout_steps = rollout_steps
        self.topk = topk
        self.topk_dist = torch.ones(ensemble_dim) / ensemble_dim # model selection distribution
        self.rollout_min_epoch = rollout_min_epoch # used to calculate rollout steps
        self.rollout_max_epoch = rollout_max_epoch # used to calculate rollout steps
        self.termination_fn = termination_fn
        self.real_ratio = real_ratio
        self.eval_ratio = eval_ratio
        self.m_steps = m_steps

        self.reward = MLP(
            obs_dim + act_dim, 1, hidden_dim, num_hidden, activation
        )
        self.dynamics = EnsembleMLP(
            obs_dim + act_dim, obs_dim * 2, ensemble_dim, hidden_dim, num_hidden, activation
        )
        
        # set reward and dynamics decay weights
        self.decay = decay
        if self.decay is None:
            self.decay = [0. for l in self.dynamics.layers if hasattr(l, "weight")]
        
        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr
        )
        self.optimizers["dynamics"] = torch.optim.Adam(
            self.dynamics.parameters(), lr=lr, 
        )
        
        # buffer to store environment data
        self.real_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, momentum=0.9)

        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "critic_loss_avg", 
            "actor_loss_avg", "rwd_mae_avg", "obs_mae_avg"
        ]

        self.max_model_lv = nn.Parameter(torch.ones(self.obs_dim) / 2, requires_grad=False)
        self.min_model_lv = nn.Parameter(-torch.ones(self.obs_dim) * 10, requires_grad=False)

        self.obs_mean = nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(self.obs_dim), requires_grad=False)
        self.rwd_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.rwd_variance = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def update_stats(self):
        self.obs_mean.data = torch.from_numpy(self.real_buffer.obs_mean).to(torch.float32)
        self.obs_variance.data = torch.from_numpy(self.real_buffer.obs_variance).to(torch.float32)
        self.rwd_mean.data = torch.from_numpy(self.real_buffer.rwd_mean).to(torch.float32)
        self.rwd_variance.data = torch.from_numpy(self.real_buffer.rwd_variance).to(torch.float32)

    def compute_reward(self, obs, act):
        r = self.reward.forward(torch.cat([obs, act], dim=-1)).clip(-self.rwd_clip_max, self.rwd_clip_max)
        return r
    
    def compute_transition_dist(self, obs, act):
        obs_act = torch.cat([obs, act], dim=-1)
        mu, lv = torch.chunk(self.dynamics.forward(obs_act), 2, dim=-1)

        if not self.clip_lv:
            std = torch.exp(lv.clip(np.log(1e-3), np.log(3)))
        else:
            lv = self.max_model_lv - F.softplus(self.max_model_lv - lv)
            lv = self.min_model_lv + F.softplus(lv - self.min_model_lv)
            std = torch.exp(lv)
        return torch_dist.Normal(mu, std)
    
    def compute_transition_log_prob(self, obs, act, next_obs):
        return self.compute_transition_dist(obs, act).log_prob(next_obs.unsqueeze(-2))
    
    def sample_transition_dist(self, obs, act):
        next_obs_dist = self.compute_transition_dist(obs, act)
        next_obs = next_obs_dist.rsample()
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_ = ensemble_idx.unsqueeze(-1).repeat_interleave(obs.shape[-1], dim=-1) # duplicate alone feature dim
        next_obs = torch.gather(next_obs, -2, ensemble_idx_).squeeze(-2)
        return next_obs

    def compute_reward_loss(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        r = batch["rwd"]

        r_pred = self.compute_reward(obs, act)
        decay_loss = self.compute_decay_loss(self.reward)
        loss = torch.pow(r_pred - r, 2).mean() + decay_loss
        return loss

    def compute_dynamics_loss(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]

        logp = self.compute_transition_log_prob(obs, act, next_obs).sum(-1)
        decay_loss = self.compute_decay_loss(self.dynamics)
        loss = -logp.mean() + decay_loss
        return loss
    
    def compute_decay_loss(self, model):
        i, loss = 0, 0
        for layer in model.layers:
            if hasattr(layer, "weight"):
                loss += self.decay[i] * torch.sum(layer.weight ** 2) / 2.
                i += 1
        return loss

    def eval_reward(self, batch):
        self.reward.eval()

        obs = batch["obs"]
        act = batch["act"]
        rwd = batch["rwd"]
        
        with torch.no_grad():
            rwd_pred = self.compute_reward(obs, act)
        
        rwd_mae = torch.abs(rwd_pred - rwd).mean()
        
        stats = {
            "rwd_mae": rwd_mae.data.item()
        }
        return stats

    def eval_model(self, batch):
        self.dynamics.eval()

        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]

        with torch.no_grad():
            next_obs_pred = self.compute_transition_dist(obs, act).mean
        
        obs_mae = torch.abs(next_obs_pred - next_obs.unsqueeze(-2)).mean((0, 2))
        
        stats = {f"obs_mae_{i}": obs_mae[i].data.item() for i in range(self.ensemble_dim)}
        stats["obs_mae"] = obs_mae.mean().data.item()
        return stats
    
    def take_reward_gradient_step(self, batch):
        self.reward.train()
        
        reward_loss = self.compute_reward_loss(batch)
        reward_loss.backward()
        self.optimizers["reward"].step()
        self.optimizers["reward"].zero_grad()
        
        stats = {
            "rwd_loss": reward_loss.data.item(),
        }
        self.reward.eval()
        return stats

    def take_model_gradient_step(self, batch):
        self.dynamics.train()
        
        dynamics_loss = self.compute_dynamics_loss(batch)
        dynamics_loss.backward()
        self.optimizers["dynamics"].step()
        self.optimizers["dynamics"].zero_grad()
        
        stats = {
            "obs_loss": dynamics_loss.data.item(),
        }

        self.dynamics.eval()
        return stats

    def train_model_epoch(self, steps, logger=None, verbose=False):
        # train test split
        num_eval = int(self.eval_ratio * self.real_buffer.size)
        data = self.real_buffer.sample(self.real_buffer.size)

        # normalize data
        data["obs"] = normalize(data["obs"], self.obs_mean, self.obs_variance)
        data["next_obs"] = normalize(data["next_obs"], self.obs_mean, self.obs_variance)
        data["rwd"] = normalize(data["rwd"], self.rwd_mean, self.rwd_variance)

        train_data = {k:v[:-num_eval] for k, v in data.items()}
        eval_data = {k:v[-num_eval:] for k, v in data.items()}
        
        best_loss = 1e6
        epoch_since_last_update = 0
        stats_epoch = {f"obs_mae_{i}": 1e6 for i in range(self.ensemble_dim)}
        for e in range(steps):
            # shuffle train data
            idx_train = np.arange(len(train_data["obs"]))
            np.random.shuffle(idx_train)

            train_stats_epoch = []
            for i in range(0, train_data["obs"].shape[0], self.batch_size):
                idx_batch = idx_train[i:i+self.batch_size]
                train_batch = {k:v[idx_batch] for k, v in train_data.items()}

                reward_train_stats = self.take_reward_gradient_step(train_batch)
                model_train_stats = self.take_model_gradient_step(train_batch)

                train_stats_epoch.append({**reward_train_stats, **model_train_stats})
            
            # evaluate
            reward_eval_stats = self.eval_reward(eval_data)
            model_eval_stats = self.eval_model(eval_data)
            
            # log stats
            train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()
            eval_stats_epoch = {**reward_eval_stats, **model_eval_stats}
            stats_epoch = {**train_stats_epoch, **eval_stats_epoch}
            if logger is not None:
                logger.push(stats_epoch)

            # termination condition based on eval performance
            current_loss = stats_epoch["rwd_mae"] + stats_epoch["obs_mae"]
            improvement = (best_loss - current_loss) / best_loss
            best_loss = min(best_loss, current_loss)
            if improvement > 0.01:
                epoch_since_last_update = 0
            else:
                epoch_since_last_update += 1
            
            if epoch_since_last_update > 5:
                break

            if verbose:
                print("e", e + 1, {k: round(v, 4) for k, v in stats_epoch.items()})

        return stats_epoch

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
            policy_stats_epoch.append(policy_stats)

            if logger is not None:
                logger.push(policy_stats)

        policy_stats_epoch = pd.DataFrame(policy_stats_epoch).mean(0).to_dict()
        return policy_stats_epoch
    
    def rollout_dynamics(self, obs, done, rollout_steps):
        """ Rollout dynamics model

        Args:
            obs (torch.tensor): observations. size=[batch_size, obs_dim]
            done (torch.tensor): done flag. size=[batch_size, 1]
            rollout_steps (int): number of rollout steps

        Returns:
            data (dict): size=[rollout_steps, batch_size, dim]
        """
        self.reward.eval()
        self.dynamics.eval()
        
        obs0 = obs.clone()
        obs = obs.clone()
        done = done.clone()
        data = {"obs": [], "act": [], "next_obs": [], "rwd": [], "done": []}
        for t in range(rollout_steps):
            with torch.no_grad():
                act = self.choose_action(obs)

                obs_norm = normalize(obs, self.obs_mean, self.obs_variance)
                rwd_norm = self.compute_reward(obs_norm, act)
                next_obs_norm = self.sample_transition_dist(obs_norm, act)

                rwd = denormalize(rwd_norm, self.rwd_mean, self.rwd_variance)
                next_obs = denormalize(next_obs_norm, self.obs_mean, self.obs_variance)

            if self.termination_fn is not None:
                done = self.termination_fn(obs.numpy(), act.numpy(), next_obs.numpy())
                done = torch.from_numpy(done).view(-1, 1).to(torch.float32)
            
            data["obs"].append(obs)
            data["act"].append(act)
            data["next_obs"].append(next_obs)
            data["rwd"].append(rwd)
            data["done"].append(done)
            
            obs = next_obs.clone()
            
            # # reinit samples
            # if self.termination_fn is not None:
            #     idx = torch_dist.Categorical(self.topk_dist).sample((int(done.sum()), ))
            #     obs[done.flatten() == 1] = obs0[idx]

        data["obs"] = torch.stack(data["obs"])
        data["act"] = torch.stack(data["act"])
        data["next_obs"] = torch.stack(data["next_obs"])
        data["rwd"] = torch.stack(data["rwd"])
        data["done"] = torch.stack(data["done"])
        return data
    
    def sample_imagined_data(self, batch_size, rollout_steps, mix=True):
        """ Sample model rollout data and add to replay buffer
        
        Args:
            batch_size (int): rollout batch size
            rollout_steps (int): model rollout steps
            mix (bool, optional): whether to mix real and fake initial states
        """
        if not mix:
            batch = self.real_buffer.sample(batch_size)
        else:
            real_batch = self.real_buffer.sample(int(batch_size/2))
            fake_batch = self.replay_buffer.sample(int(batch_size/2))
            batch = {
                real_k: torch.cat([real_v, fake_v], dim=0) 
                for ((real_k, real_v), (fake_k, fake_v)) 
                in zip(real_batch.items(), fake_batch.items())
            }
        
        rollout_data = self.rollout_dynamics(batch["obs"], batch["done"], rollout_steps)
        rollout_data = {k: v.flatten(0, 1) for k, v in rollout_data.items()}
        self.replay_buffer.push_batch(
            rollout_data["obs"].numpy(),
            rollout_data["act"].numpy(),
            rollout_data["rwd"].numpy(),
            rollout_data["next_obs"].numpy(),
            rollout_data["done"].numpy()
        )
    
    def compute_rollout_steps(self, epoch):
        """ Linearly increate rollout steps based on epoch """
        min_rollout_steps = 1
        ratio = (epoch - self.rollout_min_epoch) / (self.rollout_max_epoch - self.rollout_min_epoch)
        rollout_steps = min(
            self.rollout_steps, max(
                min_rollout_steps, min_rollout_steps + ratio * (self.rollout_steps - min_rollout_steps)
            )
        )
        return int(rollout_steps)

    def compute_topk_dist(self, stats):
        """ Compute top k model selection distribution """
        maes = [v for k, v in stats.items() if "obs_mae_" in k]
        idx_topk = np.argsort(maes)[:self.topk]
        topk_dist = np.zeros(self.ensemble_dim)
        topk_dist[idx_topk] = 1./self.topk
        self.topk_dist = torch.from_numpy(topk_dist).to(torch.float32)
        print("top k dist", self.topk_dist.numpy())

    def train_policy(
        self, env, eval_env, max_steps, epochs, steps_per_epoch, update_after, 
        update_model_every, update_policy_every,
        rwd_fn=None, num_eval_eps=0, callback=None, verbose=True
        ):
        logger = Logger()

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
        obs, eps_return, eps_len = env.reset()[0], 0, 0
        for t in range(total_steps):
            if (t + 1) < update_after:
                act = torch.rand(self.act_dim).uniform_(-1, 1) * self.act_lim
                act = act.data.numpy()
            else:
                with torch.no_grad():
                    act = self.choose_action(
                        torch.from_numpy(obs).view(1, -1).to(torch.float32)
                    ).numpy().flatten()
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

            # train model
            if (t + 1) >= update_after and (t - update_after + 1) % update_model_every == 0:
                self.update_stats()                
                model_stats_epoch = self.train_model_epoch(self.m_steps, logger=logger)
                self.compute_topk_dist(model_stats_epoch)
                if verbose:
                    round_loss_dict = {k: round(v, 3) for k, v in model_stats_epoch.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")
                
                # generate imagined data
                self.replay_buffer.clear()
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.sample_imagined_data(
                    self.rollout_batch_size, rollout_steps, mix=False
                )
                print("epoch", epoch + 1, "buffer size", self.real_buffer.size, "rollout steps", rollout_steps)

            # train policy
            if (t + 1) > update_after and (t - update_after + 1) % update_policy_every == 0:
                policy_stats_epoch = self.train_policy_epoch(logger=logger)
                if (t + 1) % verbose == 0:
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
        
        env.close()
        return logger