import numpy as np

import gym
from gym import spaces

# default LQR parameters for 2d navigation
dt = 0.1
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B = np.array([
    [dt**2, 0],
    [0, dt**2],
    [1, 0],
    [0, 1]
])
I = np.diag(2. * np.array([1, 1, 1, 1]))
Q = np.diag(np.array([10., 10., 2., 2.]))
R = np.diag(np.array([1., 1.]))

# initial state distribution
MU = np.array([100., 100., 0., 0.])
SIGMA = np.array([50., 50., 15., 15.])

class LQR(gym.Env):
    """ Linear quadratic gaussian environment. Default parameters is 2d navigation """
    def __init__(
        self, mu=MU, sigma=SIGMA, A=A, B=B, I=I, Q=Q, R=R):
        """
        Args:
            state_dim (int): state dimension
            act_dim (int): action_dimension
            mu (np.array): initial state mean. size=[state_dim]
            sigma (np.array): initial state std. size=[state_dim]
            A (np.array): transition matrix A. size=[state_dim, state_dim]
            B (np.array): control matrix B. size=[state_dim, act_dim]
            I (np.array): noise covariance matrix I. size=[state_dim, state_dim]
            Q (np.array): state cost matrix Q. size=[state_dim, state_dim]
            R (np.array): control cost matrix R. size=[act_dim, act_dim]
        """
        self.state_dim = A.shape[-1]
        self.act_dim = B.shape[-1]
        self.observation_space = spaces.Box(-float("inf"), float("inf"), shape=(self.state_dim,))
        self.action_space = spaces.Box(-float("inf"), float("inf"), shape=(self.act_dim,))
        
        self.mu = mu
        self.sigma = sigma
        self.A = A
        self.B = B
        self.I = I
        self.Q = Q
        self.R = R

    def reset(self):
        self.s = np.random.normal(self.mu, self.sigma)
        return self.s

    def step(self, a):
        w = np.random.normal(size=(1, self.state_dim)).dot(self.I)
        s_next = self.A.dot(self.s.reshape(-1, 1)) + self.B.dot(a.reshape(-1, 1)) + w.T
        
        c = self.compute_cost(self.s, a)
        self.s = s_next.flatten().copy()
        terminated = False
        return self.s, float(c), terminated, {}

    def compute_cost(self, s, a):
        s_ = s.reshape(-1, 1)
        a_ = a.reshape(-1, 1)
        c = 0.5 * (s_.T.dot(self.Q).dot(s_) + a_.T.dot(self.R).dot(a_))
        return c

if __name__ == "__main__":
    np.random.seed(0)
    
    state_dim = 4
    act_dim = 2
    
    # test env
    env = LQR()

    obs = env.reset()
    next_obs, r, _, _ = env.step(np.array([1., 1.]))

    assert list(obs.shape) == [state_dim]
    assert list(next_obs.shape) == [state_dim]
    assert isinstance(r, float)
    print("lqr env passed")

    # test vectorized env
    env = gym.vector.AsyncVectorEnv([
        lambda: LQR(),
        lambda: LQR()
    ])
    
    obs = env.reset()
    next_obs, r, _, _ = env.step(np.array([[1., 1.], [-1., -1.]]))
    
    assert list(obs.shape) == [2, state_dim]
    assert list(next_obs.shape) == [2, state_dim]
    assert len(r) == 2
    print("vectorized lqr env passed")