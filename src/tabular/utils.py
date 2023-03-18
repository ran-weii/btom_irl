import numpy as np
import torch

def rollout(env, agent, max_steps, truncate=False):
    s = env.reset()
    data = {"s": [s], "a": [], "r": []}
    for t in range(max_steps):
        with torch.no_grad():
            a = agent.choose_action(
                torch.from_numpy(np.array([s])).to(torch.float32)
            ).numpy()[0]
        s, r, done, info = env.step(a)
        
        if truncate and done:
            break

        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)

    data["s"] = np.stack(data["s"])
    data["a"] = np.stack(data["a"])
    data["r"] = np.stack(data["r"])
    return data

def rollout_parallel(env, agent, max_steps, truncate=False):
    s = env.reset()
    data = {"s": [s], "a": [], "r": []}
    for t in range(max_steps):
        with torch.no_grad():
            a = agent.choose_action(
                torch.from_numpy(s).to(torch.float32)
            ).numpy()
        s, r, done, info = env.step(a)
        
        if truncate and all(done):
            break

        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)

    data["s"] = np.stack(data["s"])
    data["a"] = np.stack(data["a"])
    data["r"] = np.stack(data["r"])
    return data

def entropy(p, eps=1e-6):
    is_numpy = False
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
        is_numpy = True

    logp = torch.log(p + eps)
    ent = -torch.sum(p * logp, dim=-1)

    if is_numpy:
        ent = ent.data.numpy()
    return ent

def kl_divergence(p, q, eps=1e-6):
    is_numpy = False
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
        q = torch.from_numpy(q)
        is_numpy = True

    logp = torch.log(p + eps)
    logq = torch.log(q + eps)
    kl = torch.sum(p * (logp - logq), dim=-1)

    if is_numpy:
        kl = kl.data.numpy()
    return kl

def compute_state_marginal(data, state_dim):
    s = data["s"][:, :-1].flatten()
    s_unique, counts = np.unique(s, return_counts=True)

    p = np.zeros((state_dim,))
    p[s_unique] = counts
    p = p / p.sum()
    return p

def compute_state_action_marginal(data, state_dim, act_dim):
    s = data["s"][:, :-1].flatten()
    a = data["a"].flatten()

    p = np.zeros((state_dim, act_dim))
    for i in range(act_dim):
        for j in range(state_dim):
            idx = np.stack([a == i, s == j]).all(0)
            p[j, i] += idx.sum()
    p /= p.sum()
    return p

def compute_mle_init_dist(data, state_dim):
    """ Compute maximum likelihood initial state distribution """
    s0, counts = np.unique(data["s"][:, 0], return_counts=True)
    init_dist = np.zeros((state_dim,))
    init_dist[s0] += counts
    init_dist /= init_dist.sum()
    return init_dist

def compute_mle_transition(data, state_dim, act_dim):
    """ Compute maximum likelihood transition matrix """
    s = data["s"][:, :-1].flatten()
    a = data["a"].flatten()
    s_next = data["s"][:, 1:].flatten()

    transition = np.zeros((act_dim, state_dim, state_dim)) + 1e-6
    for i in range(act_dim):
        for j in range(state_dim):
            idx = np.stack([a == i, s == j]).all(0)
            s_next_a = s_next[idx]
            if len(s_next_a) > 0:
                s_next_unique, count = np.unique(s_next_a, return_counts=True)
                transition[i, j, s_next_unique] += count

    transition = transition / transition.sum(-1, keepdims=True)
    return transition