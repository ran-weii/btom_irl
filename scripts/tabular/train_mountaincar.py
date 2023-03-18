import argparse
import os
import datetime
import json
import pandas as pd
import pickle
import numpy as np
import torch

from src.tabular.discrete_agent import DiscreteAgent
from src.tabular.discrete_btom import DiscreteBTOM
from src.tabular.discrete_mceirl import DiscreteMCEIRL
from src.env.mountain_car import DiscreteMountainCar
from src.tabular.utils import compute_mle_transition, compute_state_marginal

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../../data/mountain_car")
    parser.add_argument("--num_traj", type=int, default=100)
    # env args
    parser.add_argument("--x_bins", type=int, default=20, help="number of position grids, default=20")
    parser.add_argument("--v_bins", type=int, default=20, help="number of velocity grids, default=20")
    # agent args
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor, default=0.9")
    parser.add_argument("--alpha", type=float, default=1., help="softmax temperature, default=1.")
    parser.add_argument("--horizon", type=int, default=0, help="planning horizon, 0 for infinite horizon, default=0")
    # algo args
    parser.add_argument("--algo", type=str, choices=["btom", "irl", "pil"], default="btom")
    parser.add_argument("--fit_transition", type=bool_, default=True)
    parser.add_argument("--fit_reward", type=bool_, default=True)
    parser.add_argument("--mle_transition", type=bool_, default=True, help="init with mle transition if fit_transition")
    parser.add_argument("--pess_penalty", type=float, default=0., help="pessimistic penalty, default=0.")
    parser.add_argument("--rollout_steps", type=int, default=30, help="number of rollout steps, default=30")
    parser.add_argument("--exact", type=bool_, default=True, help="whether to perform exact computation, default=True")
    parser.add_argument("--obs_penalty", type=float, default=1., help="transition likelihood penalty, default=1.")
    parser.add_argument("--lr", type=float, default=0.05, help="adam learning rate, default=0.05")
    parser.add_argument("--decay", type=float, default=0., help="adam weight decay, default=0.")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs, default=10")
    parser.add_argument("--seed", type=int, default=0)
    # save args
    parser.add_argument("--exp_path", type=str, default="../../exp")
    parser.add_argument("--save", type=bool_, default=True)

    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    print(f"{date_time}, training mountain car agent using setting: {arglist}\n")
    
    # load data
    data_path = arglist["data_path"]
    with open(os.path.join(data_path, "data.p"), "rb") as f:
        data = pickle.load(f) 
    data = {k: v[:arglist["num_traj"]] for k, v in data.items()}

    print("data size", data["s"].shape)
    
    # get true params
    env = DiscreteMountainCar(arglist["x_bins"], arglist["v_bins"])
    true_transition = torch.from_numpy(env.transition_matrix).to(torch.float32)
    true_reward = torch.from_numpy(env.reward).to(torch.float32)
    
    # get empirical params
    state_dim = env.state_dim
    act_dim = env.act_dim
    mle_transition = compute_mle_transition(data, state_dim, act_dim)
    mle_transition = torch.from_numpy(mle_transition).to(torch.float32)
    state_marginal = compute_state_marginal(data, state_dim)
    state_marginal = torch.from_numpy(state_marginal).to(torch.float32)

    # init agent
    gamma = arglist["gamma"]
    alpha = arglist["alpha"]
    horizon = arglist["horizon"]
    agent = DiscreteAgent(state_dim, act_dim, gamma, alpha, horizon)
    
    # init model
    if arglist["algo"] == "btom":
        model = DiscreteBTOM(
            agent, 
            arglist["rollout_steps"],
            exact=arglist["exact"],
            obs_penalty=arglist["obs_penalty"], 
            lr=arglist["lr"], 
            decay=arglist["decay"]
        )
    elif arglist["algo"] in ["irl", "pil"]:
        arglist["fit_transition"] = False
        arglist["fit_reward"] = True
        arglist["pess_penalty"] = 0 if arglist["algo"] == "irl" else arglist["pess_penalty"]
        model = DiscreteMCEIRL(
            agent, 
            state_marginal,
            arglist["rollout_steps"],
            pess_penalty=arglist["pess_penalty"],
            exact=arglist["exact"],
            lr=arglist["lr"], 
            decay=arglist["decay"]
        )
    
    # load parameters
    if not arglist["fit_transition"]:
        if arglist["mle_transition"]:
            model.agent.log_transition.data = torch.log(mle_transition + 1e-6)
            print("loaded mle transition")
        else:
            model.agent.log_transition.data = torch.log(true_transition + 1e-6)
            print("loaded true transition")
        model.agent.log_transition.requires_grad = False
    if not arglist["fit_reward"]:
        model.agent.log_target.data = true_reward
        model.agent.log_target.requires_grad = False
        print("loaded true reward")
    
    history = model.fit(data, arglist["epochs"])

    # save model
    if arglist["save"]:
        save_path = os.path.join(
            arglist["exp_path"],
            "mountain_car",
            arglist["algo"], 
            date_time
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
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