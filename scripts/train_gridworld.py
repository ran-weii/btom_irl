import argparse
import os
import datetime
import json
import pandas as pd
import pickle
import numpy as np
import torch

from src.agents.discrete_agents import DiscreteAgent
from src.algo.btom import DiscreteBTOM

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../data/gridworld")
    # env args
    parser.add_argument("--num_grids", type=int, default=5, help="number of grids, default=5")
    # agent args
    parser.add_argument("--gamma", type=float, default=0.7, help="discount factor, default=0.7")
    parser.add_argument("--alpha", type=float, default=1., help="softmax temperature, default=1.")
    parser.add_argument("--horizon", type=int, default=0, help="planning horizon, 0 for infinite horizon, default=0")
    # algo args
    parser.add_argument("--algo", type=str, choices=["btom"], default="btom")
    parser.add_argument("--rollout_steps", type=int, default=30, help="number of rollout steps, default=30")
    parser.add_argument("--exact", type=bool_, default=True, help="whether to perform exact computation, default=True")
    parser.add_argument("--obs_penalty", type=float, default=1., help="transition likelihood penalty, default=1.")
    parser.add_argument("--lr", type=float, default=0.05, help="adam learning rate, default=0.05")
    parser.add_argument("--decay", type=float, default=0., help="adam weight decay, default=0.")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs, default=10")
    parser.add_argument("--seed", type=int, default=0)
    # save args
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--save", type=bool_, default=True)

    arglist = vars(parser.parse_args())
    return arglist

def get_mle_init_dist(data, state_dim):
    s0, counts = np.unique(data["s"][:, 0], return_counts=True)
    init_dist = np.zeros((state_dim,))
    init_dist[s0] += counts
    init_dist /= init_dist.sum()
    return init_dist

def get_mle_transition(data, state_dim, act_dim):
    s = data["s"][:, :-1].flatten()
    a = data["a"].flatten()
    s_next = data["s"][:, 1:].flatten()

    transition = np.zeros((act_dim, state_dim, state_dim)) + 1e-6
    for i in range(act_dim):
        for j in range(state_dim):
            idx = np.stack([a == i, s == j]).all(0)
            s_next_a = s_next[idx]
            if len(s_next_a) > 0:
                s_next_unique, count = np.unique(s_next_a, return_counts=True)
                transition[i, j, s_next_unique] += count

    transition = transition / transition.sum(-1, keepdims=True)
    return transition

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    print(f"{date_time}, training gridworld agent using setting: {arglist}\n")
    
    # load data
    data_path = arglist["data_path"]
    with open(os.path.join(data_path, "data.p"), "rb") as f:
        data = pickle.load(f)
    
    # init estimates
    state_dim = int(arglist["num_grids"] ** 2)
    act_dim = 5
    transition = get_mle_transition(data, state_dim, act_dim)

    # init agent
    gamma = arglist["gamma"]
    alpha = arglist["alpha"]
    horizon = arglist["horizon"]
    agent = DiscreteAgent(state_dim, act_dim, gamma, alpha, horizon)
    
    # init model
    model = DiscreteBTOM(
        agent, 
        arglist["rollout_steps"],
        exact=arglist["exact"],
        obs_penalty=arglist["obs_penalty"], 
        lr=arglist["lr"], 
        decay=arglist["decay"]
    )
    history = model.fit(data, arglist["epochs"])
    
    print(pd.DataFrame(history))

    # save model
    if arglist["save"]:
        exp_path = arglist["exp_path"]
        env_path = os.path.join(exp_path, "gridworld")
        algo_path = os.path.join(env_path, arglist["algo"])
        save_path = os.path.join(algo_path, date_time)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(env_path):
            os.mkdir(env_path)
        if not os.path.exists(algo_path):
            os.mkdir(algo_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(arglist, f)
        
        # save history
        df_history = pd.DataFrame(history)
        df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)

        # save model
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer_state_dict = {
            k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in model.optimizer.state_dict().items()
        }
        torch.save({
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }, os.path.join(save_path, "model.pt"))

        print(f"\nmodel saved at: {save_path}\n")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)