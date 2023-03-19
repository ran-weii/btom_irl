import numpy as np
import gymnasium as gym
from src.utils.data import normalize, denormalize

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


class Termination:
    def __init__(self, obs_mean=0., obs_variance=1.):
        self.obs_mean = obs_mean
        self.obs_variance = obs_variance

    def termination_fn(self, obs, act, next_obs):
        raise NotImplementedError
    

class HopperTermination(Termination):
    def __init__(self, obs_mean=0., obs_variance=1.):
        super().__init__(obs_mean, obs_variance)
    
    def termination_fn(self, obs, act, next_obs):
        next_obs = denormalize(next_obs.copy(), self.obs_mean, self.obs_variance)
        
        height = next_obs[..., 0]
        angle = next_obs[..., 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[..., 1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        return done


class HalfCheetahTermination(Termination):
    def __init__(self, obs_mean=0., obs_variance=1.):
        super().__init__(obs_mean, obs_variance)

    def termination_fn(self, obs, act, next_obs):
        next_obs = denormalize(next_obs.copy(), self.obs_mean, self.obs_variance)

        not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
        done = ~not_done
        return done
    

class WalkerTermination(Termination):
    def __init__(self, obs_mean=0., obs_variance=1.):
        super().__init__(obs_mean, obs_variance)

    def termination_fn(self, obs, act, next_obs):
        next_obs = denormalize(next_obs.copy(), self.obs_mean, self.obs_variance)

        height = next_obs[..., 0]
        angle = next_obs[..., 1]
        not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                    * (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        return done
    

class AntTermination(Termination):
    def __init__(self, obs_mean=0., obs_variance=1.):
        super().__init__(obs_mean, obs_variance)

    def termination_fn(self, obs, act, next_obs):
        next_obs = denormalize(next_obs.copy(), self.obs_mean, self.obs_variance)

        x = next_obs[..., 0]
        not_done = 	np.isfinite(next_obs).all(axis=-1) \
                    * (x >= 0.2) \
                    * (x <= 1.0)

        done = ~not_done
        return done


def get_termination_fn(env_name, obs_mean=0., obs_variance=1.):
    if "Hopper" in env_name:
        termination_fn = HopperTermination(obs_mean, obs_variance).termination_fn
    if "HalfCheetah" in env_name:
        termination_fn = HalfCheetahTermination(obs_mean, obs_variance).termination_fn
    if "Walker" in env_name:
        termination_fn = WalkerTermination(obs_mean, obs_variance).termination_fn
    if "Ant" in env_name:
        termination_fn = AntTermination(obs_mean, obs_variance).termination_fn
    return termination_fn