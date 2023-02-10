import gymnasium as gym
from src.agents.rl_utils import normalize

class GymEnv(gym.Env):
    """ Gym wrapper with normalization """
    def __init__(self, env_name, obs_mean=0., obs_std=1., **kwargs):
        self.env = gym.make(env_name, **kwargs)

        self.obs_mean = obs_mean
        self.obs_std = obs_std


    def reset(self):
        obs, info = self.env.reset()
        obs = normalize(obs, self.obs_mean, self.obs_std)
        return obs, info

    def step(self, act):
        next_obs, r, done, truncated, info = self.env.step(act)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_std)
        return next_obs, r, done, truncated, info

    def close(self):
        self.env.close()