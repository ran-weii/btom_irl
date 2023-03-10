import pprint
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_value=0):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {k: pad_sequence([b[k] for b in batch], padding_value=pad_value) for k in keys}
    mask = pad_sequence([torch.ones(len(b[keys[0]])) for b in batch])
    return pad_batch, mask

def parse_stacked_trajectories(obs, act, rwd, next_obs, terminated, timeout, max_eps=None):
    eps_id = np.cumsum(terminated + timeout)
    eps_id = np.insert(eps_id, 0, 0)[:-1] # offset by 1 step
    max_eps = eps_id.max() + 1 if max_eps is None else max_eps

    dataset = []
    for e in np.unique(eps_id):
        dataset.append({
            "obs": obs[eps_id == e],
            "act": act[eps_id == e],
            "rwd": rwd[eps_id == e],
            "next_obs": next_obs[eps_id == e],
            "done": terminated[eps_id == e],
        })

        if (e + 1) >= max_eps:
            break
    return dataset

def update_moving_stats(x, old_mean, old_mean_square, old_variance, size, momentum):
    """ Compute moving mean and variance stats from batch data """
    batch_size = len(x)
    new_mean = (old_mean * size + np.sum(x, axis=0)) / (size + batch_size)
    new_mean_square = (old_mean_square * size + np.sum(x**2, axis=0)) / (size + batch_size)
    new_variance = new_mean_square - new_mean**2
    
    # print(old_mean, size)
    # print(x.mean(0))
    # print("new mean", new_mean)

    new_mean = old_mean * momentum + new_mean * (1 - momentum)
    new_mean_square = old_mean_square * momentum + new_mean_square * (1 - momentum)
    new_variance = old_variance * momentum + new_variance * (1 - momentum)
    return new_mean, new_mean_square, new_variance

def normalize(x, mean, variance):
    return (x - mean) / variance**0.5

def denormalize(x, mean, variance):
    return x * variance**0.5 + mean


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
        self.obs_batch = np.concatenate([self.obs_batch, obs.reshape(1, -1)], axis=0)
        self.act_batch = np.concatenate([self.act_batch, act.reshape(1, -1)], axis=0)
        self.rwd_batch = np.concatenate([self.rwd_batch, rwd.reshape(1, -1)], axis=0)
        self.next_obs_batch = np.concatenate([self.next_obs_batch, next_obs.reshape(1, -1)], axis=0)
        self.done_batch = np.concatenate([self.done_batch, done.reshape(1, -1)], axis=0)

    def push_batch(self, obs=None, act=None, rwd=None, next_obs=None, done=None):
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

        batch_size = len(obs)

        self.obs = np.concatenate([self.obs, obs.reshape(-1, self.obs_dim)], axis=0)
        self.act = np.concatenate([self.act, act.reshape(-1, self.act_dim)], axis=0)
        self.rwd = np.concatenate([self.rwd, rwd.reshape(-1, 1)], axis=0)
        self.next_obs = np.concatenate([self.next_obs, next_obs.reshape(-1, self.obs_dim)], axis=0)
        self.done = np.concatenate([self.done, done.reshape(-1, 1)], axis=0)

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
    
    def sample_episodes(self, batch_size):
        return 

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
    
    def sample_episodes(self, batch_size, prioritize=False, ratio=2):
        """ Sample complete episodes with zero sequence padding 

        Args:
            batch_size (int): sample batch size.
            prioritize (bool, optional): whether to perform prioritized sampling. Default=False
            ratio (int, optional): prioritization ratio. 
                Sample from the latest batch_size * ratio episodes. Deafult=100
        """
        if prioritize:
            max_samples = min(self.num_eps, batch_size * ratio)
            idx = np.random.choice(np.arange(max_samples), min(batch_size, max_samples), replace=False)
        else:
            idx = np.random.choice(np.arange(self.num_eps), min(batch_size, self.num_eps), replace=False)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.obs[i]).to(torch.float32)
            act = torch.from_numpy(self.act[i]).to(torch.float32)
            rwd = torch.from_numpy(self.rwd[i]).to(torch.float32)
            next_obs = torch.from_numpy(self.next_obs[i]).to(torch.float32)
            done = torch.from_numpy(self.done[i]).to(torch.float32)
            
            batch.append({
                "obs": obs, 
                "act": act, 
                "rwd": rwd, 
                "next_obs": next_obs,
                "done": done,
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


class Logger():
    """ Reinforcement learning stats logger """
    def __init__(self):
        self.epoch_dict = dict()
        self.history = []
        self.test_episodes = []
    
    def push(self, stats_dict):
        for key, val in stats_dict.items():
            if not (key in self.epoch_dict.keys()):
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(val)

    def log(self):
        stats = dict()
        for key, val in self.epoch_dict.items():
            if isinstance(val[0], np.ndarray) or len(val) > 1:
                vals = np.stack(val)
                stats[key + "_avg"] = np.mean(vals)
                stats[key + "_std"] = np.std(vals)
                stats[key + "_min"] = np.min(vals)
                stats[key + "_max"] = np.max(vals)
            else:
                stats[key] = val[-1]

        pprint.pprint({k: np.round(v, 4) for k, v, in stats.items()})
        self.history.append(stats)

        # erase epoch stats
        self.epoch_dict = dict()