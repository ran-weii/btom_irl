import argparse
import os
import datetime
import json
import pandas as pd
import pickle
import numpy as np
import torch

from src.agents.lqr_agent import LQRAgent
from src.algo.lqr_btom import LQRBTOM

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../data/lqr")
    # agent args
    parser.add_argument("--gamma", type=float, default=0.7, help="discount factor, default=0.7")
    parser.add_argument("--alpha", type=float, default=1., help="softmax temperature, default=1.")
    # algo args
    parser.add_argument("--algo", type=str, choices=["btom"], default="btom")
    parser.add_argument("--rollout_steps", type=int, default=30, help="number of rollout steps, default=30")
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

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    print(f"{date_time}, training lqr agent using setting: {arglist}\n")
    
    # load data
    data_path = arglist["data_path"]
    with open(os.path.join(data_path, "data.p"), "rb") as f:
        data = pickle.load(f)
        
    # init agent
    state_dim = data["s"].shape[-1]
    act_dim = data["a"].shape[-1]
    gamma = arglist["gamma"]
    alpha = arglist["alpha"]
    horizon = 0
    agent = LQRAgent(state_dim, act_dim, gamma, alpha, horizon)
    
    # init model
    model = LQRBTOM(
        agent, 
        arglist["rollout_steps"],
        obs_penalty=arglist["obs_penalty"], 
        lr=arglist["lr"], 
        decay=arglist["decay"]
    )
    history = model.fit(data, arglist["epochs"])

    # save model
    if arglist["save"]:
        exp_path = arglist["exp_path"]
        env_path = os.path.join(exp_path, "lqr")
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