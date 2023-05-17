import numpy as np
import torch
from src.utils.data import update_moving_stats, collate_fn

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size, momentum=0.1):
        """ Replay buffer to store independent samples, episodes cannot be retrieved

        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            max_size (int): maximum buffer size
            momentum (float, optional): moving stats momentum. Default=0.99
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = 0
        self.max_size = max_size
        self.momentum = momentum

        self.obs = np.empty(shape=(0, obs_dim))
        self.act = np.empty(shape=(0, act_dim))
        self.rwd = np.empty(shape=(0, 1))
        self.next_obs = np.empty(shape=(0, obs_dim))
        self.done = np.empty(shape=(0, 1))
        
        # batch placeholder
        self.obs_batch = np.empty(shape=(0, obs_dim))
        self.act_batch = np.empty(shape=(0, act_dim))
        self.rwd_batch = np.empty(shape=(0, 1))
        self.next_obs_batch = np.empty(shape=(0, obs_dim))
        self.done_batch = np.empty(shape=(0, 1))
        
        # moving stats
        self.obs_mean = np.zeros((obs_dim,))
        self.obs_mean_square = np.zeros((obs_dim,))
        self.obs_variance = np.ones((obs_dim, ))

        self.rwd_mean = np.zeros((1,))
        self.rwd_mean_square = np.zeros((1,))
        self.rwd_variance = np.ones((1,))
    
    def clear(self):
        self.obs = np.empty(shape=(0, self.obs_dim))
        self.act = np.empty(shape=(0, self.act_dim))
        self.rwd = np.empty(shape=(0, 1))
        self.next_obs = np.empty(shape=(0, self.obs_dim))
        self.done = np.empty(shape=(0, 1))
        
        # batch placeholder
        self.obs_batch = np.empty(shape=(0, self.obs_dim))
        self.act_batch = np.empty(shape=(0, self.act_dim))
        self.rwd_batch = np.empty(shape=(0, 1))
        self.next_obs_batch = np.empty(shape=(0, self.obs_dim))
        self.done_batch = np.empty(shape=(0, 1))

        self.size = 0

    def push(self, obs, act, rwd, next_obs, done):
        """ Temporarily store """
        self.obs_batch = np.concatenate([self.obs_batch, obs.reshape(-1, self.obs_dim)], axis=0)
        self.act_batch = np.concatenate([self.act_batch, act.reshape(-1, self.act_dim)], axis=0)
        self.rwd_batch = np.concatenate([self.rwd_batch, rwd.reshape(-1, 1)], axis=0)
        self.next_obs_batch = np.concatenate([self.next_obs_batch, next_obs.reshape(-1, self.obs_dim)], axis=0)
        self.done_batch = np.concatenate([self.done_batch, done.reshape(-1, 1)], axis=0)

    def push_batch(self, obs=None, act=None, rwd=None, next_obs=None, done=None, update_stats=True):
        assert (
            all([obs is None, act is None, rwd is None, next_obs is None, done is None]) or 
            all([obs is not None, act is not None, rwd is not None, next_obs is not None, done is not None])
        )

        if obs is None:
            obs = self.obs_batch
            act = self.act_batch
            rwd = self.rwd_batch
            next_obs = self.next_obs_batch
            done = self.done_batch

            self.obs_batch = np.empty(shape=(0, self.obs_dim))
            self.act_batch = np.empty(shape=(0, self.act_dim))
            self.rwd_batch = np.empty(shape=(0, 1))
            self.next_obs_batch = np.empty(shape=(0, self.obs_dim))
            self.done_batch = np.empty(shape=(0, 1))
        
        assert done.dtype != bool
        
        batch_size = len(obs)

        self.obs = np.concatenate([self.obs, obs.reshape(-1, self.obs_dim)], axis=0)
        self.act = np.concatenate([self.act, act.reshape(-1, self.act_dim)], axis=0)
        self.rwd = np.concatenate([self.rwd, rwd.reshape(-1, 1)], axis=0)
        self.next_obs = np.concatenate([self.next_obs, next_obs.reshape(-1, self.obs_dim)], axis=0)
        self.done = np.concatenate([self.done, done.reshape(-1, 1)], axis=0)
        
        if update_stats:
            self.update_stats(obs, rwd)
        self.size += batch_size

        if self.size > self.max_size:
            size_diff = int(self.size - self.max_size)
            
            self.obs = self.obs[size_diff:]
            self.act = self.act[size_diff:]
            self.rwd = self.rwd[size_diff:]
            self.next_obs = self.next_obs[size_diff:]
            self.done = self.done[size_diff:]

            self.size -= size_diff

    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        idx = np.random.choice(np.arange(self.size), batch_size, replace=False)

        batch = dict(
            obs=self.obs[idx], 
            act=self.act[idx], 
            rwd=self.rwd[idx], 
            next_obs=self.next_obs[idx], 
            done=self.done[idx],
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}

    def update_stats(self, obs, rwd):
        """ Update observation and reward moving mean and variance """
        self.obs_mean, self.obs_mean_square, self.obs_variance = update_moving_stats(
            obs, self.obs_mean, self.obs_mean_square, self.obs_variance, self.size, self.momentum
        )

        self.rwd_mean, self.rwd_mean_square, self.rwd_variance = update_moving_stats(
            rwd, self.rwd_mean, self.rwd_mean_square, self.rwd_variance, self.size, self.momentum
        )


class EpisodeReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size, momentum=0.99):
        """ Replay buffer to store full episodes

        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            max_size (int): maximum buffer size
            momentum (float, optional): moving stats momentum. Default=0.99
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eps_len = []
        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        self.momentum = momentum

        self.obs_mean = np.zeros((obs_dim,))
        self.obs_mean_square = np.zeros((obs_dim,))
        self.obs_variance = np.ones((obs_dim, ))

        self.rwd_mean = np.zeros((1,))
        self.rwd_mean_square = np.zeros((1,))
        self.rwd_variance = np.ones((1,))

        # placeholder for all episodes
        self.obs = []
        self.act = []
        self.rwd = []
        self.next_obs = []
        self.done = []
        
        # placeholder for a single episode
        self.obs_eps = [] # store a single episode
        self.act_eps = [] # store a single episode
        self.next_obs_eps = [] # store a single episode
        self.rwd_eps = [] # store a single episode
        self.done_eps = [] # store a single episode

    def __call__(self, obs, act, rwd, next_obs, done):
        """ Append transition to episode placeholder """ 
        self.obs_eps.append(obs)
        self.act_eps.append(act)
        self.next_obs_eps.append(next_obs)
        self.rwd_eps.append(np.array(rwd).reshape(1, 1))
        self.done_eps.append(np.array([int(done)]).reshape(1, 1))
    
    def clear(self):
        self.obs = []
        self.act = []
        self.rwd = []
        self.next_obs = []
        self.done = []

        self.eps_len = []
        self.num_eps = 0
        self.size = 0
        
    def push(self, obs=None, act=None, rwd=None, next_obs=None, done=None):
        """ Store episode data to buffer """
        if obs is None and act is None:
            obs = np.vstack(self.obs_eps)
            act = np.vstack(self.act_eps)
            next_obs = np.vstack(self.next_obs_eps)
            rwd = np.vstack(self.rwd_eps)
            done = np.vstack(self.done_eps)            

        # stack episode at the top of the buffer
        self.obs.insert(0, obs)
        self.act.insert(0, act)
        self.rwd.insert(0, rwd)
        self.next_obs.insert(0, next_obs)
        self.done.insert(0, done)

        self.eps_len.insert(0, len(obs))
        self.update_stats(obs, rwd)

        # update self size
        self.num_eps += 1
        self.size += len(obs)
        if self.size > self.max_size:
            while self.size > self.max_size:
                self.obs = self.obs[:-1]
                self.act = self.act[:-1]
                self.rwd = self.rwd[:-1]
                self.next_obs = self.next_obs[:-1]
                self.done = self.done[:-1]

                self.size -= len(self.obs[-1])
                self.eps_len = self.eps_len[:-1]
                self.num_eps = len(self.eps_len)
        
        # clear episode
        self.obs_eps = []
        self.act_eps = []
        self.rwd_eps = []
        self.next_obs_eps = []
        self.done_eps = []

    def sample(self, batch_size, prioritize=False, ratio=100):
        """ Sample random transitions 
        
        Args:
            batch_size (int): sample batch size.
            prioritize (bool, optional): whether to perform prioritized sampling. Default=False
            ratio (int, optional): prioritization ratio. 
                Sample from the latest batch_size * ratio transitions. Deafult=100
        """ 
        obs = np.vstack(self.obs)
        act = np.vstack(self.act)
        rwd = np.vstack(self.rwd)
        next_obs = np.vstack(self.next_obs)
        done = np.vstack(self.done)
        
        # prioritize new data for sampling
        if prioritize:
            max_samples = min(self.size, batch_size * ratio)
            idx = np.random.choice(np.arange(max_samples), min(batch_size, max_samples), replace=False)
        else:
            idx = np.random.choice(np.arange(self.size), min(batch_size, self.size), replace=False)
        
        batch = dict(
            obs=obs[idx], 
            act=act[idx], 
            rwd=rwd[idx], 
            next_obs=next_obs[idx], 
            done=done[idx],
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}
    
    def sample_episode_segments(self, batch_size, seq_len=1000, prioritize=False, ratio=2):
        """ Sample episode segments with zero sequence padding 

        Args:
            batch_size (int): sample batch size.
            seq_len (int, optional): sequence length. Default=1000
            prioritize (bool, optional): whether to perform prioritized sampling. Default=False
            ratio (int, optional): prioritization ratio. 
                Sample from the latest batch_size * ratio episodes. Deafult=100
        """
        if prioritize:
            max_samples = min(self.num_eps, batch_size * ratio)
            idx = np.random.choice(np.arange(max_samples), min(batch_size, max_samples), replace=False)
        else:
            # idx = np.random.choice(np.arange(self.num_eps), min(batch_size, self.num_eps), replace=False)
            idx = np.random.randint(0, self.num_eps, batch_size)

        batch = []
        for i in idx:
            obs = torch.from_numpy(self.obs[i]).to(torch.float32)
            act = torch.from_numpy(self.act[i]).to(torch.float32)
            rwd = torch.from_numpy(self.rwd[i]).to(torch.float32)
            next_obs = torch.from_numpy(self.next_obs[i]).to(torch.float32)
            done = torch.from_numpy(self.done[i]).to(torch.float32)
            
            t = np.random.randint(0, len(obs) - seq_len)

            batch.append({
                "obs": obs[t:t+seq_len], 
                "act": act[t:t+seq_len], 
                "rwd": rwd[t:t+seq_len], 
                "next_obs": next_obs[t:t+seq_len],
                "done": done[t:t+seq_len],
            })
        
        out = collate_fn(batch)
        return out

    def update_stats(self, obs, rwd):
        """ Update observation and reward moving mean and variance """
        self.obs_mean, self.obs_mean_square, self.obs_variance = update_moving_stats(
            obs, self.obs_mean, self.obs_mean_square, self.obs_variance, self.size, self.momentum
        )

        self.rwd_mean, self.rwd_mean_square, self.rwd_variance = update_moving_stats(
            rwd, self.rwd_mean, self.rwd_mean_square, self.rwd_variance, self.size, self.momentum
        )