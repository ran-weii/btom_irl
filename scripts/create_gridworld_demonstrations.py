import argparse
import os
import pickle
import gym
import numpy as np
import torch

from src.env.gridworld import Gridworld
from src.agents.discrete_agent import DiscreteAgent
from src.algo.utils import rollout_parallel

def parse_args():
    parser = argparse.ArgumentParser()
    # env args
    parser.add_argument("--num_grids", type=int, default=5, help="number of grids, default=5")
    parser.add_argument("--epsilon", type=float, default=0., help="transition error probability, default=0.1")
    # agent args
    parser.add_argument("--gamma", type=float, default=0.7, help="discount factor, default=0.7")
    parser.add_argument("--alpha", type=float, default=1., help="softmax temperature, default=1.")
    parser.add_argument("--horizon", type=int, default=0, help="agent planning horizon, 0 for infinite horizon default=0")
    # rollout args
    parser.add_argument("--num_eps", type=int, default=100, help="number of demonstration episodes, default=10")
    parser.add_argument("--max_steps", type=int, default=100, help="max number of steps in demonstration episodes, default=100")
    parser.add_argument("--num_workers", type=int, default=2, help="number of rollout workers, defualt=2")
    parser.add_argument("--seed", type=int, default=0)
    # save args
    parser.add_argument("--save_path", type=str, default="../data")
    parser.add_argument("--save", type=bool, default=True)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    print(f"generating gridworld demonstrations with setting: {arglist}")
    
    num_grids = arglist.num_grids
    init_pos = np.array([])
    goal_pos = np.array([
        [0, 4], [4, 0], [4, 4]
    ])
    epsilon = arglist.epsilon

    env = Gridworld(
        num_grids, init_pos, goal_pos, epsilon
    )

    gamma = arglist.gamma
    alpha = arglist.alpha
    horizon = arglist.horizon
    transition = torch.from_numpy(env.transition_matrix).to(torch.float32)
    target = torch.from_numpy(env.target_dist).to(torch.float32)
    log_transition = torch.log(transition + 1e-6)
    log_target = torch.log(target + 1e-6)

    agent = DiscreteAgent(env.state_dim, env.act_dim, gamma, alpha, horizon)
    agent.log_transition.data = log_transition
    agent.log_target.data = log_target
    agent.plan()

    # rollout vectorized envs
    env = gym.vector.AsyncVectorEnv([
        lambda: Gridworld(
            num_grids, init_pos=init_pos, goal_pos=goal_pos, epsilon=epsilon
        ) for i in range(arglist.num_workers)
    ])
    
    data = []
    for _ in range(arglist.num_eps // arglist.num_workers):
        data.append(rollout_parallel(env, agent, arglist.max_steps))

    data_s = np.hstack([d["s"] for d in data]).T
    data_a = np.hstack([d["a"] for d in data]).T
    data_r = np.hstack([d["r"] for d in data]).T
    data = {"s": data_s, "a": data_a, "r": data_r}
    
    print("data shape s: {}, a: {}, r: {}".format(data["s"].shape, data["a"].shape, data["r"].shape))
    print(f"average return: {data['r'].sum(1).mean():.4f}")

    # save
    if arglist.save:
        data_path = os.path.join(arglist.save_path, "gridworld")
        filename = os.path.join(data_path, "data.p")

        if not os.path.exists(arglist.save_path):
            os.mkdir(arglist.save_path)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"data saved at: {filename}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)