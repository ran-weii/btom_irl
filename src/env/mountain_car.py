import numpy as np
from gym import spaces
from gym.envs.classic_control import MountainCarEnv

class DiscreteMountainCar(MountainCarEnv):
    """ Discretized mountain car environment """
    def __init__(self, x_bins=20, v_bins=20, transition_samples=10000):
        """
        Args:
            x_bins (int, optional): number of position bins. Default=20
            v_bins (int, optional): number of velocity bins. Default=20
            transition_samples (int, optional): number of samples to estimate transition matrix. Default=10000
        """
        super().__init__()
        self.x_bins = x_bins
        self.v_bins = v_bins
        self.state_dim = x_bins * v_bins
        self.act_dim = 3
        self.eps = 1e-6
        
        obs_min = self.observation_space.low
        obs_max = self.observation_space.high
        self.x_grid = np.linspace(obs_min[0], obs_max[0], self.x_bins)
        self.v_grid = np.linspace(obs_min[1], obs_max[1], self.v_bins)
        
        # make reward: any position > goal position has 0, else -1
        goal_position = self.goal_position
        goal_velocity = self.observation_space.low[1]
        goal_obs = np.array([[goal_position, goal_velocity]])
        goal_state = self.obs2state(goal_obs)
        self.reward = -np.ones((self.state_dim,))
        self.reward[goal_state[0]:] = 0

        self.make_transition_matrix(transition_samples)
        
        self.observation_space = spaces.Discrete(self.state_dim)
        self.action_space = spaces.Discrete(self.act_dim)
    
    def reset(self):
        obs = super().reset()
        s = self.obs2state(obs)[0]
        self.obs = obs
        return s
    
    def step(self, act):
        obs, _, done, info = super().step(act)
        s = self.obs2state(obs)[0]
        rwd = self.reward[s]
        self.obs = obs
        return s, rwd, done, info
    
    def batch_step(self, state, action):
        """ Batch apply dynamics 
        
        Args:
            state (np.array): [batch_size, 2]
            action (np.array): [batch_size]
        """
        assert len(list(action.shape)) == 1
        position = state[:, 0].copy()
        velocity = state[:, 1].copy()
        velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # handle min position
        is_invalid = np.stack([position <= self.min_position, velocity < 0]).T
        is_valid = np.all(is_invalid, axis=1) == False
        velocity *= is_valid
        
        next_state = np.stack([position, velocity]).T
        return next_state
    
    def compute_reward(self, state):
        terminated = (state[:, 0] >= self.goal_position)
        return -1 * (terminated == False) + 0 * terminated

    def obs2state(self, obs):
        obs = obs.reshape(-1, 2).copy()
        d_x = (self.high[0] - self.low[0]) / self.x_bins
        d_v = (self.high[1] - self.low[1]) / self.v_bins
        x_grid = np.clip((obs[:, 0] - self.low[0]) // d_x, 0, self.x_bins - 1)
        v_grid = np.clip((obs[:, 1] - self.low[1]) // d_v, 0, self.v_bins - 1)
        state = x_grid * self.x_bins + v_grid
        return state.astype(int)

    def state2obs(self, state):
        assert np.all(state <= self.state_dim)
        d_x = (self.high[0] - self.low[0]) / (self.x_bins - 1)
        d_v = (self.high[1] - self.low[1]) / (self.v_bins - 1)

        x_grid = state // self.v_bins
        v_grid = state - x_grid * self.v_bins
        
        position = x_grid * d_x + self.low[0]
        velocity = v_grid * d_v + self.low[1]

        obs = np.stack([position, velocity]).T
        return obs

    def make_transition_matrix(self, num_samples=8000):
        """ Create discrete transition marix """
        # sample the observation space
        position = np.random.uniform(self.low[0], self.high[0], num_samples)
        velocity = np.random.uniform(self.low[1], self.high[1], num_samples)
        obs = np.stack([position, velocity]).T
        state = self.obs2state(obs)
        
        transition_matrix = self.eps * np.ones((self.act_dim, self.state_dim, self.state_dim))
        for a in range(self.act_dim):
            action = a * np.ones((num_samples,))
            next_obs = self.batch_step(obs, action)
            next_state = self.obs2state(next_obs)
            for s in range(self.state_dim):
                unique_next_state, counts = np.unique(
                    next_state[state == s], return_counts=True
                )
                transition_matrix[a, s, unique_next_state] += counts

        transition_matrix /= transition_matrix.sum(-1, keepdims=True)
        self.transition_matrix = transition_matrix

if __name__ == "__main__":
    np.random.seed(0)
    x_bins = 20
    v_bins = 20
    env = DiscreteMountainCar(x_bins, v_bins)
    
    # test transition matrix normalization
    assert np.isclose(env.transition_matrix.sum(-1), 1., atol=1e-5).all()

    # test discretizations
    obs1 = np.array([-1.2, -0.07])
    obs2 = np.array([-1.2, 0.07])
    obs3 = np.array([0.6, -0.07])
    obs4 = np.array([0.6, 0.07])
    obs5 = np.array([-1.2, -0.06])
    
    assert env.obs2state(obs1) == 0
    assert env.obs2state(obs2) == 19
    assert env.obs2state(obs3) == 380
    assert env.obs2state(obs4) == 399
    assert env.obs2state(obs5) == 1

    assert np.isclose(env.state2obs(env.obs2state(obs1)), np.array([-1.2, -0.07]), atol=1e-4).all()
    assert np.isclose(env.state2obs(env.obs2state(obs2)), np.array([-1.2, 0.07]), atol=1e-4).all()
    assert np.isclose(env.state2obs(env.obs2state(obs3)), np.array([0.6, -0.07]), atol=1e-4).all()
    assert np.isclose(env.state2obs(env.obs2state(obs4)), np.array([0.6, 0.07]), atol=1e-4).all()
    assert np.isclose(env.state2obs(env.obs2state(obs5)), np.array([-1.2, -0.062631]), atol=1e-4).all()

    print("DiscretMountainCar passed")