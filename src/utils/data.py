import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def load_data(filepath, num_samples):
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    
    # unpack dataset
    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"].reshape(-1, 1)
    next_obs = dataset["next_observations"]
    terminated = dataset["terminals"].reshape(-1, 1)
    
    # subsample data
    num_samples = min(num_samples, len(obs))
    idx = np.arange(len(obs))
    np.random.shuffle(idx)
    idx = idx[:num_samples]

    obs = obs[idx]
    act = act[idx]
    rwd = rwd[idx]
    next_obs = next_obs[idx]
    terminated = terminated[idx]
    return obs, act, rwd, next_obs, terminated

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

    new_mean = old_mean * momentum + new_mean * (1 - momentum)
    new_mean_square = old_mean_square * momentum + new_mean_square * (1 - momentum)
    new_variance = old_variance * momentum + new_variance * (1 - momentum)
    return new_mean, new_mean_square, new_variance

def normalize(x, mean, variance):
    return (x - mean) / variance**0.5

def denormalize(x, mean, variance):
    return x * variance**0.5 + mean