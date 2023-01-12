import argparse
import os
import pickle
import gym
import numpy as np
import torch

from src.env.lqr import LQR
from src.agents.lqr_agent import LQRAgent
from src.algo.utils import rollout_parallel

def parse_args():
    parser = argparse.ArgumentParser()
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
    print(f"generating lqr demonstrations with setting: {arglist}")

    env = LQR()
    
    state_dim = env.state_dim
    act_dim = env.act_dim
    gamma = arglist.gamma
    alpha = arglist.alpha
    horizon = arglist.horizon
    
    agent = LQRAgent(state_dim, act_dim, gamma, alpha, horizon)
    
    agent._A.data = torch.from_numpy(env.A).to(torch.float32)
    agent._B.data = torch.from_numpy(env.B).to(torch.float32)
    agent.log_I.data = torch.from_numpy(env.I**0.5).diagonal().log().to(torch.float32)
    agent.log_Q.data = torch.from_numpy(env.Q).diagonal().log().to(torch.float32)
    agent.log_R.data = torch.from_numpy(env.R).diagonal().log().to(torch.float32)
    with torch.no_grad():
        agent.plan()

    # rollout vectorized envs
    env = gym.vector.AsyncVectorEnv([
        lambda: LQR() for i in range(arglist.num_workers)
    ])
    
    data = []
    for _ in range(arglist.num_eps // arglist.num_workers):
        data.append(rollout_parallel(env, agent, arglist.max_steps))
    
    data_s = np.vstack([d["s"].swapaxes(0, 1) for d in data])
    data_a = np.vstack([d["a"].swapaxes(0, 1) for d in data])
    data_r = np.vstack([d["r"].swapaxes(0, 1) for d in data])
    data = {"s": data_s, "a": data_a, "r": data_r}
    
    print("data shape s: {}, a: {}, r: {}".format(data["s"].shape, data["a"].shape, data["r"].shape))
    print(f"average return: {data['r'].sum(1).mean():.4f}")
    
    # save
    if arglist.save:
        data_path = os.path.join(arglist.save_path, "lqr")
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