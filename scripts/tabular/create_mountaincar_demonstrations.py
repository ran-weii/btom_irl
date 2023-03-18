import argparse
import os
import pickle
import gym
import numpy as np
import torch

from src.env.mountain_car import DiscreteMountainCar
from src.tabular.discrete_agent import DiscreteAgent
from src.tabular.utils import rollout

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    # env args
    parser.add_argument("--x_bins", type=int, default=20, help="number of position grids, default=20")
    parser.add_argument("--v_bins", type=int, default=20, help="number of velocity grids, default=20")
    # agent args
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor, default=0.9")
    parser.add_argument("--alpha", type=float, default=10., help="softmax temperature, default=10.")
    parser.add_argument("--horizon", type=int, default=0, help="agent planning horizon, 0 for infinite horizon default=0")
    # rollout args
    parser.add_argument("--num_eps", type=int, default=100, help="number of demonstration episodes, default=10")
    parser.add_argument("--max_steps", type=int, default=200, help="max number of steps in demonstration episodes, default=200")
    parser.add_argument("--seed", type=int, default=0)
    # save args
    parser.add_argument("--save_path", type=str, default="../../data")
    parser.add_argument("--save", type=bool_, default=True)
    
    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"generating mountain car demonstrations with setting: {arglist}")
    
    # init env
    env = DiscreteMountainCar(arglist["x_bins"], arglist["v_bins"])

    gamma = arglist["gamma"]
    alpha = arglist["alpha"]
    horizon = arglist["horizon"]
    transition = torch.from_numpy(env.transition_matrix).to(torch.float32)
    reward = torch.from_numpy(env.reward).to(torch.float32)
    log_transition = torch.log(transition + 1e-6)
    log_target = reward

    agent = DiscreteAgent(env.state_dim, env.act_dim, gamma, alpha, horizon)
    agent.log_transition.data = log_transition
    agent.log_target.data = log_target
    agent.plan()

    # rollout envs
    data = []
    for _ in range(arglist["num_eps"]):
        data.append(rollout(env, agent, arglist["max_steps"]))

    data_s = np.stack([d["s"] for d in data])
    data_a = np.stack([d["a"] for d in data])
    data_r = np.stack([d["r"] for d in data])
    data = {"s": data_s, "a": data_a, "r": data_r}
    
    print("data shape s: {}, a: {}, r: {}".format(data["s"].shape, data["a"].shape, data["r"].shape))
    print(f"average return: {data['r'].sum(1).mean():.4f}")

    # save
    if arglist["save"]:
        data_path = os.path.join(arglist["save_path"], "mountain_car")
        filename = os.path.join(data_path, "data.p")
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"data saved at: {filename}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)