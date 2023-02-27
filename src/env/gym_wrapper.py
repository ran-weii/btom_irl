import numpy as np
import gymnasium as gym
from src.agents.rl_utils import normalize, denormalize

class GymEnv(gym.Env):
    """ Gym wrapper with normalization """
    def __init__(self, env_name, obs_mean=0., obs_variance=1., rwd_mean=0., rwd_variance=1., **kwargs):
        self.env = gym.make(env_name, **kwargs)

        self.obs_mean = obs_mean
        self.obs_variance = obs_variance

        self.rwd_mean = rwd_mean
        self.rwd_variance = rwd_variance

    def reset(self):
        obs, info = self.env.reset()
        obs = normalize(obs.copy(), self.obs_mean, self.obs_variance)
        return obs, info

    def step(self, act):
        next_obs, r, done, truncated, info = self.env.step(act)
        next_obs = normalize(next_obs.copy(), self.obs_mean, self.obs_variance)
        return next_obs, r, done, truncated, info

    def close(self):
        self.env.close()


class HopperTermination:
    def __init__(self, obs_mean=0., obs_std=1.):
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        
        next_obs = denormalize(next_obs.copy(), self.obs_mean, self.obs_std)
        
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        return done


def get_termination_fn(env_name, obs_mean=0., obs_std=1.):
    if "Hopper" in env_name:
        termination_fn = HopperTermination(obs_mean, obs_std).termination_fn
    return termination_fn