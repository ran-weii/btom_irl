import time
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

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size, momentum=0.1):
        """ Replay buffer for fully observable environments

        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            max_size (int): maximum buffer size
            momentum (float, optional): moving stats momentum. Default=0.1
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # self.episodes = []
        self.eps_len = []

        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        
        self.momentum = momentum
        self.moving_mean = np.zeros((obs_dim,))
        self.moving_mean_square = np.zeros((obs_dim,))
        self.moving_variance = np.ones((obs_dim, ))

        # placeholder for all episodes
        self.obs = []
        self.absorb = []
        self.act = []
        self.rwd = []
        self.next_obs = []
        self.next_absorb = []
        self.done = []
        self.truncated = []
        
        # placeholder for a single episode
        self.obs_eps = [] # store a single episode
        self.act_eps = [] # store a single episode
        self.next_obs_eps = [] # store a single episode
        self.rwd_eps = [] # store a single episode
        self.done_eps = [] # store a single episode
        self.truncated_eps = [] # store a single episode

    def __call__(self, obs, act, next_obs, rwd, done, truncated):
        """ Append transition to episode placeholder """ 
        self.obs_eps.append(obs)
        self.act_eps.append(act)
        self.next_obs_eps.append(next_obs)
        self.rwd_eps.append(np.array(rwd).reshape(1, 1))
        self.done_eps.append(np.array([int(done)]).reshape(1, 1))
        self.truncated_eps.append(np.array([int(truncated)]).reshape(1, 1))
    
    def clear(self):
        # self.episodes = []

        self.obs = []
        self.absorb = []
        self.act = []
        self.rwd = []
        self.next_obs = []
        self.next_absorb = []
        self.done = []
        self.truncated = []

        self.eps_len = []
        self.num_eps = 0
        self.size = 0
        
    def push(self, obs=None, act=None, next_obs=None, rwd=None, done=None, truncated=None):
        """ Store episode data to buffer """
        if obs is None and act is None:
            obs = np.vstack(self.obs_eps)
            act = np.vstack(self.act_eps)
            next_obs = np.vstack(self.next_obs_eps)
            rwd = np.vstack(self.rwd_eps)
            done = np.vstack(self.done_eps)
            truncated = np.vstack(self.truncated_eps)

        absorb = np.zeros((len(obs), 1))
        next_absorb = np.zeros((len(obs), 1))

        # stack episode at the top of the buffer
        self.obs.insert(0, obs)
        self.absorb.insert(0, absorb)
        self.act.insert(0, act)
        self.rwd.insert(0, rwd)
        self.next_obs.insert(0, next_obs)
        self.next_absorb.insert(0, next_absorb)
        self.done.insert(0, done)
        self.truncated.insert(0, truncated)

        self.eps_len.insert(0, len(obs))
        self.update_obs_stats(obs)

        # update self size
        self.num_eps += 1
        self.size += len(obs)
        if self.size > self.max_size:
            while self.size > self.max_size:
                self.obs = self.obs[:-1]
                self.absorb = self.absorb[:-1]
                self.act = self.act[:-1]
                self.rwd = self.rwd[:-1]
                self.next_obs = self.next_obs[:-1]
                self.next_absorb = self.next_absorb[:-1]
                self.done = self.done[:-1]
                self.truncated = self.truncated[:-1]

                self.size -= len(self.obs[-1])
                self.eps_len = self.eps_len[:-1]
                self.num_eps = len(self.eps_len)
        
        # clear episode
        self.obs_eps = []
        self.act_eps = [] 
        self.next_obs_eps = []
        self.rwd_eps = []
        self.done_eps = []
        self.truncated_eps = []

    def sample(self, batch_size, prioritize=False, ratio=100):
        """ Sample random transitions 
        
        Args:
            batch_size (int): sample batch size.
            prioritize (bool, optional): whether to perform prioritized sampling. Default=False
            ratio (int, optional): prioritization ratio. 
                Sample from the latest batch_size * ratio transitions. Deafult=100
        """ 
        obs = np.vstack(self.obs)
        absorb = np.vstack(self.absorb)
        act = np.vstack(self.act)
        rwd = np.vstack(self.rwd)
        next_obs = np.vstack(self.next_obs)
        next_absorb = np.vstack(self.next_absorb)
        done = np.vstack(self.done)
        truncated = np.vstack(self.truncated)
        
        # prioritize new data for sampling
        if prioritize:
            max_samples = min(self.size, batch_size * ratio)
            idx = np.random.choice(np.arange(max_samples), min(batch_size, max_samples), replace=False)
        else:
            idx = np.random.choice(np.arange(self.size), min(batch_size, self.size), replace=False)
        
        batch = dict(
            obs=obs[idx], 
            absorb=absorb[idx],
            act=act[idx], 
            rwd=rwd[idx], 
            next_obs=next_obs[idx], 
            next_absorb=next_absorb[idx],
            done=done[idx],
            truncated=truncated[idx]
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}
    
    def sample_episodes(self, batch_size, prioritize=False, ratio=2, sample_truncated=False):
        """ Sample complete episodes with zero sequence padding 

        Args:
            batch_size (int): sample batch size.
            prioritize (bool, optional): whether to perform prioritized sampling. Default=False
            ratio (int, optional): prioritization ratio. 
                Sample from the latest batch_size * ratio episodes. Deafult=100
            sample_truncated (bool, optional): whether to sample truncated episodes. Default=False
        """
        if prioritize:
            max_samples = min(self.num_eps, batch_size * ratio)
            idx = np.random.choice(np.arange(max_samples), min(batch_size, max_samples), replace=False)
        else:
            idx = np.random.choice(np.arange(self.num_eps), min(batch_size, self.num_eps), replace=False)
        
        batch = []
        for i in idx:
            obs = torch.from_numpy(self.obs[i]).to(torch.float32)
            absorb = torch.from_numpy(self.absorb[i]).to(torch.float32)
            act = torch.from_numpy(self.act[i]).to(torch.float32)
            rwd = torch.from_numpy(self.rwd[i]).to(torch.float32)
            next_obs = torch.from_numpy(self.next_obs[i]).to(torch.float32)
            next_absorb = torch.from_numpy(self.next_absorb[i]).to(torch.float32)
            done = torch.from_numpy(self.done[i]).to(torch.float32)
            truncated = torch.from_numpy(self.truncated[i]).to(torch.float32)
            
            if not sample_truncated and truncated.sum().data.item() > 0:
                continue
            
            batch.append({
                "obs": obs, 
                "absorb": absorb,
                "act": act, 
                "rwd": rwd, 
                "next_obs": next_obs,
                "next_absorb": next_absorb, 
                "done": done,
                "truncated": truncated
            })
        
        out = collate_fn(batch)
        return out

    def update_obs_stats(self, obs):
        """ Update observation moving mean and variance """
        batch_size = len(obs)
        
        moving_mean = (self.moving_mean * self.size + np.sum(obs, axis=0)) / (self.size + batch_size)
        moving_mean_square = (self.moving_mean_square * self.size + np.sum(obs**2, axis=0)) / (self.size + batch_size)
        moving_variance = moving_mean_square - moving_mean**2

        self.moving_mean = self.moving_mean * (1 - self.momentum) + moving_mean * self.momentum
        self.moving_mean_square = self.moving_mean_square * (1 - self.momentum) + moving_mean_square * self.momentum
        self.moving_variance = self.moving_variance * (1 - self.momentum) + moving_variance * self.momentum 


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


def train(
    env, model, epochs, max_steps=500, steps_per_epoch=1000, 
    update_after=3000, update_every=50, verbose=False, callback=None, 
    ):
    """ RL training loop adapted from: Openai spinning up
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
    
    Args:
        env (gym.Env): simulator environment
        model (Model): trainer model
        epochs (int): training epochs
        max_steps (int, optional): maximum environment steps before done. Default=500
        steps_per_epoch (int, optional): number of environment steps per epoch. Default=1000
        update_after (int, optional): initial burn-in steps before training. Default=3000
        update_every (int, optional): number of environment steps between training. Default=50
        custom_reward (class, optional): custom reward function. Default=None
        verbose (bool, optional): whether to print instantaneous loss between epoch. Default=False
        callback (class, optional): a callback class. Default=None
        render (bool, optional): whether to render environment. Default=False
    """
    model.eval()
    logger = Logger()

    total_steps = epochs * steps_per_epoch
    start_time = time.time()
    
    epoch = 0
    obs, eps_return, eps_len = env.reset()[0], 0, 0
    for t in range(total_steps):
        with torch.no_grad():
            act = model.choose_action(
                torch.from_numpy(obs).view(1, -1).to(torch.float32)
            ).numpy().flatten()
        next_obs, reward, terminated, truncated, info = env.step(act)
        
        eps_return += reward
        eps_len += 1
        
        model.replay_buffer(obs, act, next_obs, reward, terminated)
        obs = next_obs
        
        # end of trajectory handeling
        if terminated or (eps_len + 1) >= max_steps:
            model.replay_buffer.push()
            logger.push({"eps_return": eps_return/eps_len})
            logger.push({"eps_len": eps_len})
            
            # start new episode
            obs, eps_return, eps_len = env.reset()[0], 0, 0

        # train model
        if t >= update_after and t % update_every == 0:
            train_stats = model.take_gradient_step(logger)

            if verbose:
                round_loss_dict = {k: round(v, 4) for k, v in train_stats.items()}
                print(f"e: {epoch}, t: {t}, {round_loss_dict}")

        # end of epoch handeling
        if (t + 1) % steps_per_epoch == 0:
            model.on_epoch_end(logger)

            epoch = (t + 1) // steps_per_epoch

            logger.push({"epoch": epoch})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

            if t > update_after and callback is not None:
                callback(model, logger)
    
    env.close()
    if callback is not None:
        callback(model, logger)
    return model, logger

def train_async(
    env, model, epochs, max_steps=500, steps_per_epoch=1000, 
    update_after=3000, update_every=50, verbose=False, callback=None
    ):
    """ Asynchronous RL training loop adapted from: Openai spinning up
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
    
    Args:
        env (gym.Env): simulator environment
        model (Model): trainer model
        epochs (int): training epochs
        max_steps (int, optional): maximum environment steps before done. Default=500
        steps_per_epoch (int, optional): number of environment steps per epoch. Default=1000
        update_after (int, optional): initial burn-in steps before training. Default=3000
        update_every (int, optional): number of environment steps between training. Default=50
        custom_reward (class, optional): custom reward function. Default=None
        verbose (bool, optional): whether to print instantaneous loss between epoch. Default=False
        callback (class, optional): a callback class. Default=None
        render (bool, optional): whether to render environment. Default=False
    """
    model.eval()
    logger = Logger()

    total_steps = epochs * steps_per_epoch
    start_time = time.time()
    
    temp_buffer = {"obs": [], "act": [], "rwd": [], "next_obs": [], "terminated": [], "truncated": []}
    epoch = 0
    obs, eps_return, eps_len = env.reset()[0], 0, 0
    for t in range(total_steps):
        with torch.no_grad():
            act = model.choose_action(
                torch.from_numpy(obs).to(torch.float32)
            ).numpy()
        next_obs, reward, terminated, truncated, info = env.step(act)
        
        temp_buffer["obs"].append(obs)
        temp_buffer["act"].append(act)
        temp_buffer["rwd"].append(reward)
        temp_buffer["next_obs"].append(next_obs)
        temp_buffer["terminated"].append(terminated)
        temp_buffer["truncated"].append(truncated) # dummy
        
        eps_len += 1
        obs = next_obs
        
        # end of trajectory handeling
        if (eps_len + 1) > max_steps:
            buffer_obs = np.stack(temp_buffer["obs"])
            buffer_act = np.stack(temp_buffer["act"])
            buffer_rwd = np.stack(temp_buffer["rwd"])
            buffer_next_obs = np.stack(temp_buffer["next_obs"])
            buffer_terminated = 1 * np.stack(temp_buffer["terminated"])
            buffer_truncated = 1 * np.stack(temp_buffer["truncated"])
            
            for i in range(buffer_obs.shape[1]): # iterate workers
                worker_dataset = parse_stacked_trajectories(
                    buffer_obs[:, i], buffer_act[:, i], buffer_rwd[:, i], buffer_next_obs[:, i], 
                    buffer_terminated[:, i], buffer_truncated[:, i]
                )
                for episode in worker_dataset:
                    # manual handle truncation
                    worker_eps_len = len(episode["obs"])
                    episode["truncated"] = np.zeros((worker_eps_len, 1))

                    """ todo: maybe drop truncated episodes """
                    if sum(episode["done"]) == 0 and worker_eps_len < max_steps:
                        episode["truncated"][-1, 0] = 1
                    else:
                        logger.push({"eps_return": episode["rwd"].sum()})
                        logger.push({"eps_len": worker_eps_len})
                        
                        print("worker eps len", worker_eps_len, "reward", episode["rwd"].sum())

                        model.replay_buffer.push(
                            episode["obs"], episode["act"], episode["next_obs"],
                            episode["rwd"].reshape(-1, 1), episode["done"].reshape(-1, 1),
                            episode["truncated"]
                        )

            # start new episode
            temp_buffer = {"obs": [], "act": [], "rwd": [], "next_obs": [], "terminated": [], "truncated": []}
            obs, eps_return, eps_len = env.reset()[0], 0, 0

            print(t, model.replay_buffer.num_eps, model.replay_buffer.size, buffer_obs.shape, "\n")

        # train model
        if (t + 1) > update_after and (t + 1) % update_every == 0:
            train_stats = model.take_gradient_step(logger)

            if verbose:
                round_loss_dict = {k: round(v, 4) for k, v in train_stats.items()}
                print(f"e: {epoch}, t: {t + 1}, {round_loss_dict}")

        # end of epoch handeling
        if (t + 1) > update_after and (t + 1) % steps_per_epoch == 0:
            model.on_epoch_end(logger)

            epoch = (t + 1) // steps_per_epoch

            logger.push({"epoch": epoch})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

            if (t + 1) > update_after and callback is not None:
                callback(model, logger)
    
    # env.close()
    if callback is not None:
        callback(model, logger)
    return model, logger
