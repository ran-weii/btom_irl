import numpy as np

def create_experiment(num_grids, init_type, goal_type, p_goal=0.5):
    """ Create initial and goal distribution specifications for gridworld 
    
    Args:
        num_grids (int): number of grids per side
        init_type (str): choices of ["one_state", "uniform"]
        goal_type (str): choices of ["one_goal", "three_goals"]
        p_goal (float, optional): goal probability of upper right corner. Default=0.5

    Returns:
        init_specs (dict): initial state specs
        goal_specs (dict): goal state specs
    """
    if init_type == "one_state":
        init_specs = {
            (0, 0): 1.
        }
    elif init_type == "uniform":
        init_specs = {}
    else:
        raise ValueError("init_type not supported, choose from ['one_state', 'uniform']")

    if goal_type == "one_goal":
        goal_specs = {
            (num_grids - 1, num_grids - 1): 1.
        } 
    elif goal_type == "three_goals":
        goal_specs = {
            (0, num_grids - 1): (1 - p_goal) / 2,
            (num_grids - 1, 0): (1 - p_goal) / 2,
            (num_grids - 1, num_grids - 1): p_goal,
        }
    else:
        raise ValueError("init_type not supported, choose from ['one_goal', 'three_goals']")
    return init_specs, goal_specs


if __name__ == "__main__":
    from src.env.gridworld import Gridworld

    def test_create_experiment():
        num_grids = 5

        init_type = "one_state"
        goal_type = "one_goal"
        init_specs, goal_specs = create_experiment(num_grids, init_type, goal_type)
        env = Gridworld(num_grids,init_specs,goal_specs)
        init_dist = env.init_dist.reshape(num_grids, num_grids)
        target_dist = env.target_dist.reshape(num_grids, num_grids)
        
        assert env.init_dist.sum(-1) == 1.
        assert env.target_dist.sum(-1) == 1.
        assert init_dist[0, 0] == 1.
        assert target_dist[num_grids - 1, num_grids - 1] == 1.

        init_type = "uniform"
        goal_type = "one_goal"
        init_specs, goal_specs = create_experiment(num_grids, init_type, goal_type)
        env = Gridworld(num_grids,init_specs,goal_specs)
        init_dist = env.init_dist.reshape(num_grids, num_grids)
        target_dist = env.target_dist.reshape(num_grids, num_grids)
        
        assert env.init_dist.sum(-1) == 1.
        assert env.target_dist.sum(-1) == 1.
        assert init_dist[0, 0] == 1 / (env.state_dim - 1)
        assert target_dist[num_grids - 1, num_grids - 1] == 1.
        
        init_type = "uniform"
        goal_type = "three_goals"
        p_goal = 0.3333333333
        init_specs, goal_specs = create_experiment(num_grids, init_type, goal_type, p_goal=p_goal)
        env = Gridworld(num_grids,init_specs,goal_specs)
        init_dist = env.init_dist.reshape(num_grids, num_grids)
        target_dist = env.target_dist.reshape(num_grids, num_grids)
        
        assert env.init_dist.sum(-1) == 1.
        assert env.target_dist.sum(-1) == 1.
        assert init_dist[0, 0] == 1 / (env.state_dim - 3)
        assert target_dist[num_grids - 1, 0] == (1 - p_goal) / 2
        assert target_dist[0, num_grids - 1] == (1 - p_goal) / 2
        assert target_dist[num_grids - 1, num_grids - 1] == p_goal
        
    test_create_experiment()
    print("test_create_experiment passed")
