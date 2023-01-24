import time
import pprint
import numpy as np
import torch

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

        self.episodes = []
        self.eps_len = []

        self.num_eps = 0
        self.size = 0
        self.max_size = max_size
        
        self.momentum = momentum
        self.moving_mean = np.zeros((obs_dim,))
        self.moving_mean_square = np.zeros((obs_dim,))
        self.moving_variance = np.ones((obs_dim, ))
        
        # placeholder for a single episode
        self.obs_eps = [] # store a single episode
        self.act_eps = [] # store a single episode
        self.next_obs_eps = [] # store a single episode
        self.rwd_eps = [] # store a single episode
        self.done_eps = [] # store a single episode

    def __call__(self, obs, act, next_obs, rwd, done=False):
        """ Append transition to episode placeholder """ 
        self.obs_eps.append(obs)
        self.act_eps.append(act)
        self.next_obs_eps.append(next_obs)
        self.rwd_eps.append(np.array(rwd).reshape(1, 1))
        self.done_eps.append(np.array([int(done)]).reshape(1, 1))
    
    def clear(self):
        self.episodes = []
        self.eps_len = []
        self.num_eps = 0
        self.size = 0
        
    def push(self, obs=None, act=None, next_obs=None, rwd=None, done=None):
        """ Store episode data to buffer """
        if obs is None and act is None:
            obs = np.vstack(self.obs_eps)
            act = np.vstack(self.act_eps)
            next_obs = np.vstack(self.next_obs_eps)
            rwd = np.vstack(self.rwd_eps)
            done = np.vstack(self.done_eps)
        
        # add absorbing state flag based on done
        if done[-1] == 1:
            obs_a = np.zeros((1, obs.shape[1]))
            act_a = np.zeros((1, act.shape[1]))
            done_a = np.ones((1, 1))
            
            # add transition (s_T -> s_a)
            obs = np.vstack([obs, next_obs[-1:]])
            act = np.vstack([act, act_a])
            next_obs = np.vstack([next_obs, obs_a])
            rwd = np.vstack([rwd, rwd[-1:]])
            done = np.vstack([done, done_a])

            # add transition (s_a -> s_a)
            obs = np.vstack([obs, obs_a])
            act = np.vstack([act, act_a])
            next_obs = np.vstack([next_obs, obs_a])
            rwd = np.vstack([rwd, rwd[-1:]])
            done = np.vstack([done, done_a])

            # add absorbing indicator
            absorb = np.zeros((len(done), 1))
            absorb[-1:] = 1
            next_absorb = np.zeros((len(done), 1))
            next_absorb[-2:] = 1
        else:
            absorb = np.zeros((len(obs), 1))
            next_absorb = np.zeros((len(obs), 1))

        self.episodes.append({ 
            "obs": obs,
            "absorb": absorb,
            "act": act,
            "rwd": rwd,
            "next_obs": next_obs,
            "next_absorb": next_absorb, 
            "done": done, # whether the next state is done
        })
        self.update_obs_stats(obs)
        
        self.eps_len.append(len(self.episodes[-1]["obs"]))
        
        # update self size
        self.num_eps += 1
        self.size += len(self.episodes[-1]["obs"])
        if self.size > self.max_size:
            while self.size > self.max_size:
                self.size -= len(self.episodes[0]["obs"])
                self.episodes = self.episodes[1:]
                self.eps_len = self.eps_len[1:]
                self.num_eps = len(self.eps_len)
        
        # clear episode
        self.obs_eps = []
        self.act_eps = [] 
        self.next_obs_eps = []
        self.rwd_eps = []
        self.done_eps = []

    def sample(self, batch_size, prioritize=False):
        """ sample random transitions 
        
        Args:
            batch_size (int): sample batch size.
            prioritize (bool, optional): whether to perform prioritized sampling. 
                If True sample the latest batch_size * 100 transitions. Default=False
        """ 
        obs = np.vstack([e["obs"] for e in self.episodes])
        absorb = np.vstack([e["absorb"] for e in self.episodes])
        act = np.vstack([e["act"] for e in self.episodes])
        rwd = np.vstack([e["rwd"] for e in self.episodes])
        next_obs = np.vstack([e["next_obs"] for e in self.episodes])
        next_absorb = np.vstack([e["next_absorb"] for e in self.episodes])
        done = np.vstack([e["done"] for e in self.episodes])
        
        # prioritize new data for sampling
        if prioritize:
            idx = np.random.randint(max(0, self.size - batch_size * 100), self.size, size=batch_size)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        
        batch = dict(
            obs=obs[idx], 
            absorb=absorb[idx],
            act=act[idx], 
            rwd=rwd[idx], 
            next_obs=next_obs[idx], 
            next_absorb=next_absorb[idx],
            done=done[idx],
        )
        return {k: torch.from_numpy(v).to(torch.float32) for k, v in batch.items()}
        
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
    update_after=3000, update_every=50, custom_reward=None,
    verbose=False, callback=None, render=False, 
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
            act = model.choose_action(torch.from_numpy(obs).to(torch.float32)).numpy()
        next_obs, reward, terminated, truncated, info = env.step(act)

        if render:
            env.render()

        if custom_reward is not None:
            reward = custom_reward(next_obs)
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
            epoch = (t + 1) // steps_per_epoch

            logger.push({"epoch": epoch})
            logger.push({"time": time.time() - start_time})
            logger.log()
            print()

            if t > update_after and callback is not None:
                callback(model, logger)
    
    env.close()
    return model, logger
