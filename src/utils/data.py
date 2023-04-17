import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

""" TODO: maybe return a dict with timeouts and add an eps_id column so that parse_stacked_traj can directly used normalized features """
def load_data(filepath, num_samples, skip_timeout=True, shuffle=True):
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    
    # unpack dataset
    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"].reshape(-1, 1)
    next_obs = dataset["next_observations"]
    terminated = 1 * dataset["terminals"].reshape(-1, 1)
    
    # follow d4rl qlearning_dataset
    if skip_timeout:
        obs = obs[dataset["timeouts"] == False]
        act = act[dataset["timeouts"] == False]
        rwd = rwd[dataset["timeouts"] == False]
        next_obs = next_obs[dataset["timeouts"] == False]
        terminated = terminated[dataset["timeouts"] == False]
    
    if shuffle:
        idx = np.arange(len(obs))
        np.random.shuffle(idx)
        
        obs = obs[idx]
        act = act[idx]
        rwd = rwd[idx]
        next_obs = next_obs[idx]
        terminated = terminated[idx]
    
    # subsample data
    num_samples = min(num_samples, len(obs))

    obs = obs[:num_samples]
    act = act[:num_samples]
    rwd = rwd[:num_samples]
    next_obs = next_obs[:num_samples]
    terminated = terminated[:num_samples]
    return obs, act, rwd, next_obs, terminated

""" TODO: figure out how to do normalization properly and not mess up in between. Should do normalization up front and not inside the agent """
def parse_stacked_trajectories(data, max_eps=None, skip_terminated=True, obs_mean=None, obs_std=None):
    obs_mean = 0. if obs_mean is None else obs_mean
    obs_std = 1. if obs_std is None else obs_std

    obs = data["observations"]
    act = data["actions"]
    rwd = data["rewards"]
    next_obs = data["next_observations"]
    terminated = data["terminals"]
    timeout = data['timeouts']

    eps_id = np.cumsum(terminated + timeout, axis=0).flatten()
    eps_id = np.insert(eps_id, 0, 0)[:-1] # offset by 1 step
    max_eps = eps_id.max() + 1 if max_eps is None else max_eps

    dataset = []
    for e in np.unique(eps_id):
        if terminated[eps_id == e].sum() > 0 and skip_terminated:
            continue

        dataset.append({
            "obs": (obs[eps_id == e] - obs_mean) / obs_std,
            "act": act[eps_id == e],
            "rwd": rwd[eps_id == e],
            "next_obs": (next_obs[eps_id == e] - obs_mean) / obs_std,
            "done": terminated[eps_id == e],
        })

        if len(dataset) >= max_eps:
            break
    return dataset

def collate_fn(batch, pad_value=0):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {k: pad_sequence([b[k] for b in batch], padding_value=pad_value) for k in keys}
    mask = pad_sequence([torch.ones(len(b[keys[0]])) for b in batch])
    return pad_batch, mask

def update_moving_stats(x, old_mean, old_mean_square, old_variance, size, momentum):
    """ Compute moving mean and variance stats from batch data 
    
    Args:
        x (np.array): new data. size=[batch_size, x_dim]
        old_mean (np.array): old mean of x. size=[x_dim]
        old_mean_square (np.array): old mean of x-squared. size=[x_dim]
        old_variance (np.array): old variance of x. size=[x_dim]
        size (int): number of old data points
        momentum (int): momentum for old stats

    Returns:
        new_mean (np.array): updated mean of x. size=[x_dim]
        new_mean_square (np.array): updated mean of x-squared. size=[x_dim]
        new_variance (np.array): updated variance of x. size=[x_dim]
    """
    batch_size = len(x)
    new_mean = (old_mean * size + np.sum(x, axis=0)) / (size + batch_size)
    new_mean_square = (old_mean_square * size + np.sum(x**2, axis=0)) / (size + batch_size)
    new_variance = new_mean_square - new_mean**2

    new_mean = old_mean * momentum + new_mean * (1 - momentum)
    new_mean_square = old_mean_square * momentum + new_mean_square * (1 - momentum)
    new_variance = old_variance * momentum + new_variance * (1 - momentum)
    return new_mean, new_mean_square, new_variance

def normalize(x, mean, variance):
    return (x - mean) / variance**0.5

def denormalize(x, mean, variance):
    return x * variance**0.5 + mean