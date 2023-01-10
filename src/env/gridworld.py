import itertools
import numpy as np

import gym
from gym import spaces

class Gridworld(gym.Env):
    """ Fully observable gridworld environment with state based reward """
    def __init__(
        self, 
        num_grids, 
        init_pos=np.array([]), 
        goal_pos=np.array([]), 
        epsilon=0.,
        init_dist=None,
        target_dist=None,
        transition_matrix=None):
        """
        Args:
            num_grids (int): number of grids per side
            init_post (np.array): initial state positions. default is uniform except goal states. size=[n, 2]
            goal_post (np.array): goal state positions. default is upper right. size=[n, 2]
            epsilon (float): transition error. Default=0.
            init_dist (torch.tensor): initial state distribution. size=[state_dim]
            target_dist (torch.tensor): target state distribution. size=[state_dim]
            transition_matrix (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
        """
        assert epsilon < 1.
        self.num_grids = num_grids
        self.state_dim = num_grids ** 2
        self.act_dim = 5
        self.a_labels = ["up", "right", "down", "left", "stay"]
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Discrete(self.state_dim)
        self.epsilon = epsilon
        
        # state to position mapping
        state2pos = list(itertools.product(np.arange(num_grids), np.arange(num_grids)))
        self.state2pos = np.array(state2pos).reshape(self.state_dim, -1).astype(int)

        # target distribution
        if len(goal_pos) == 0:
            goal_pos = np.array([[num_grids - 1, num_grids - 1]])
        self.goal_states = np.stack([self.pos2state(p) for p in goal_pos])

        self.target_dist = np.zeros(self.state_dim)
        self.target_dist[self.goal_states] = 1./len(self.goal_states)

        if target_dist is not None: # init target_dist from input
            assert isinstance(target_dist, np.ndarray)
            assert list(target_dist.shape) == [self.state_dim]
            self.target_dist = target_dist
        self.reward_matrix = np.log(self.target_dist + 1e-6)
        
        # initial state distribution
        if len(init_pos) == 0:
            init_states = np.ones(self.state_dim)
            init_states[self.goal_states] = 0.
            self.init_states = np.where(init_states == 1)[0]
        else:
            self.init_states = np.stack([self.pos2state(p) for p in init_pos])
        self.init_dist = np.zeros(self.state_dim)
        self.init_dist[self.init_states] = 1./len(self.init_states)
        
        if init_dist is not None: # init init_dist from input
            assert isinstance(init_dist, np.ndarray)
            assert list(init_dist.shape) == [self.state_dim]
            self.init_dist = init_dist
        
        if transition_matrix is not None: # init transition from input
            assert isinstance(transition_matrix, np.ndarray)
            assert list(transition_matrix.shape) == [self.act_dim, self.state_dim, self.state_dim]
            self.transition_matrix = transition_matrix
        else:
            self.make_transition_matrix()
    
    def pos2state(self, pos):
        """ Map a single position to state 
        
        Args:
            pos (torch.tensor): size=[n, 2]
        """
        return np.where(np.all(pos == self.state2pos, axis=1))[0]

    def make_transition_matrix(self):
        # pos [0, 0] is origin
        transition = np.zeros((self.act_dim, self.state_dim, self.state_dim))
        
        pos = self.state2pos.copy()
        for a in range(self.act_dim):
            next_pos = pos.copy()
            next_error_pos1 = pos.copy()
            next_error_pos2 = pos.copy()
            
            if a == 0: # up
                next_pos[:, 1] = np.clip(next_pos[:, 1] + 1, 0, self.num_grids - 1)
                next_error_pos1[:, 0] = np.clip(next_error_pos1[:, 0] - 1, 0, self.num_grids - 1)
                next_error_pos2[:, 0] = np.clip(next_error_pos2[:, 0] + 1, 0, self.num_grids - 1)
            elif a == 1: # right
                next_pos[:, 0] = np.clip(next_pos[:, 0] + 1, 0, self.num_grids - 1)
                next_error_pos1[:, 1] = np.clip(next_error_pos1[:, 1] + 1, 0, self.num_grids - 1)
                next_error_pos2[:, 1] = np.clip(next_error_pos2[:, 1] - 1, 0, self.num_grids - 1)
            elif a == 2: # down
                next_pos[:, 1] = np.clip(next_pos[:, 1] - 1, 0, self.num_grids - 1)
                next_error_pos1[:, 0] = np.clip(next_error_pos1[:, 0] - 1, 0, self.num_grids - 1)
                next_error_pos2[:, 0] = np.clip(next_error_pos2[:, 0] + 1, 0, self.num_grids - 1)
            elif a == 3: # left
                next_pos[:, 0] = np.clip(next_pos[:, 0] - 1, 0, self.num_grids - 1)
                next_error_pos1[:, 1] = np.clip(next_error_pos1[:, 1] + 1, 0, self.num_grids - 1)
                next_error_pos2[:, 1] = np.clip(next_error_pos2[:, 1] - 1, 0, self.num_grids - 1)
            elif a == 4: # stay
                pass
            
            next_states = np.hstack([self.pos2state(next_pos[i]) for i in range(len(next_pos))])
            next_error_states1 = np.hstack([self.pos2state(next_error_pos1[i]) for i in range(len(next_error_pos1))])
            next_error_states2 = np.hstack([self.pos2state(next_error_pos2[i]) for i in range(len(next_error_pos2))])
            
            # fill in transition probs
            transition[a, np.arange(self.state_dim), next_states] += 1 - self.epsilon
            transition[a, np.arange(self.state_dim), next_error_states1] += self.epsilon/2
            transition[a, np.arange(self.state_dim), next_error_states2] += self.epsilon/2
        
        self.transition_matrix = transition

    def reset(self):
        # sample initial state
        self.s = np.random.choice(np.arange(self.state_dim), p=self.init_dist)
        return self.s

    def step(self, a):
        s_dist = self.transition_matrix[a][self.s]
        s_next = np.random.choice(np.arange(self.state_dim), p=s_dist)
        r = self.reward_matrix[s_next]

        self.s = s_next
        terminated = False
        return s_next, r, terminated, {}

if __name__ == "__main__":
    np.random.seed(0)
    
    num_grids = 5
    init_pos = np.array([
        [0, 0], [0, 1], [1, 0]
    ])
    goal_pos = np.array([
        [0, 4], [4, 0], [4, 4]
    ])
    epsilon = 0.1
    
    # test env
    env = Gridworld(num_grids, init_pos=init_pos, goal_pos=goal_pos, epsilon=epsilon)

    obs = env.reset()
    next_obs, _, _, _ = env.step(1)
    
    assert env.init_dist.sum(-1) == 1
    assert env.target_dist.sum(-1) == 1
    assert np.all(env.transition_matrix.sum(-1) == 1)
    print("gridworld env passed")
    
    # test vectorized env
    env = gym.vector.AsyncVectorEnv([
        lambda: Gridworld(num_grids, init_pos=init_pos, goal_pos=goal_pos, epsilon=epsilon),
        lambda: Gridworld(num_grids, init_pos=init_pos, goal_pos=goal_pos, epsilon=epsilon)
    ])
    
    obs = env.reset()
    next_obs, _, _, _ = env.step(np.array([0, 0]))
    
    assert obs.shape[0] == 2
    assert next_obs.shape[0] == 2
    print("vectorized gridworld env passed")
    