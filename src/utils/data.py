import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence

def load_d4rl_transitions(
    dataset_name, dataset_path=None, num_samples=None, skip_timeout=True, 
    shuffle=False, norm_obs=False, norm_rwd=False
    ):
    """ Parse d4rl stacked trajectories into dictionary of transitions 
    
    Returns:
        transition_dataset (dict): dictionary of transitions
        obs_mean (np.array): observation mean
        obs_std (np.array): observation std
        rwd_mean (np.array): reward mean
        rwd_std (np.array): reward std
    """
    if dataset_path is None:
        import d4rl
        import gym
        dataset = gym.make(dataset_name).get_dataset()
    else:
        with open(dataset_path, "rb") as f:
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
    num_samples = len(obs) if num_samples is None else min(num_samples, len(obs))
    obs = obs[:num_samples]
    act = act[:num_samples]
    rwd = rwd[:num_samples]
    next_obs = next_obs[:num_samples]
    terminated = terminated[:num_samples]
    
    # normalize data
    obs_mean = 0.
    obs_std = 1.
    if norm_obs:
        obs_mean = obs.mean(0)
        obs_std = obs.std(0)
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std
    
    rwd_mean = 0.
    rwd_std = 1.
    if norm_rwd:
        rwd_mean = rwd.mean(0)
        rwd_std = rwd.std(0)
        rwd = (rwd - rwd_mean) / rwd_std
    
    print("\nprocessed data stats")
    print("obs size:", obs.shape)
    print("obs_mean:", obs.mean(0).round(2))
    print("obs_std:", obs.std(0).round(2))
    print("rwd_mean:", rwd.mean(0).round(2))
    print("rwd_std:", rwd.std(0).round(2))
    
    transition_dataset = {
        "obs": obs,
        "act": act,
        "rwd": rwd,
        "next_obs": next_obs,
        "done": terminated,
    }
    return transition_dataset, obs_mean, obs_std, rwd_mean, rwd_std

def parse_d4rl_stacked_trajectories(
    dataset_name, dataset_path=None, max_eps=None, skip_terminated=False, obs_mean=None, obs_std=None
    ):
    """ Parse d4rl stacked trajectories into list of episode dictionaries """
    if dataset_path is None:
        import d4rl
        import gym
        dataset = gym.make(dataset_name).get_dataset()
    else:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

    obs_mean = 0. if obs_mean is None else obs_mean
    obs_std = 1. if obs_std is None else obs_std

    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"]
    next_obs = dataset["next_observations"]
    terminated = dataset["terminals"]
    timeout = dataset['timeouts']

    eps_id = np.cumsum(terminated + timeout, axis=0).flatten()
    eps_id = np.insert(eps_id, 0, 0)[:-1] # offset by 1 step
    max_eps = eps_id.max() + 1 if max_eps is None else max_eps
    
    print("parsing d4rl stacked trajectories")
    traj_dataset = []
    for e in tqdm(np.unique(eps_id)):
        if terminated[eps_id == e].sum() > 0 and skip_terminated:
            continue

        traj_dataset.append({
            "obs": (obs[eps_id == e] - obs_mean) / obs_std,
            "act": act[eps_id == e],
            "rwd": rwd[eps_id == e].reshape(-1, 1),
            "next_obs": (next_obs[eps_id == e] - obs_mean) / obs_std,
            "done": 1 * terminated[eps_id == e].reshape(-1, 1),
        })

        if len(traj_dataset) >= max_eps:
            break
    return traj_dataset

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