import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.data import load_d4rl_transitions
from src.agents.dynamics import EnsembleDynamics, train_ensemble
from src.agents.buffer import ReplayBuffer
from src.utils.logger import SaveCallback, load_checkpoint

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yml_path", type=str, default="../config/dynamics/base.yml")
    parser.add_argument("--exp_path", type=str, default="../exp/mujoco/dynamics")
    parser.add_argument("--data_path", type=str, default="../data/d4rl/")
    parser.add_argument("--data_name", type=str, default="hopper-expert-v2")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # data args
    parser.add_argument("--num_samples", type=int, default=100000, help="number of training transitions, default=100000")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="train test split ratio, default=0.2")
    parser.add_argument("--norm_obs", type=bool_, default=True, help="normalize observations, default=True")
    parser.add_argument("--norm_rwd", type=bool_, default=False, help="normalize reward, default=False")
    # model args
    parser.add_argument("--pred_rwd", type=bool_, default=True, help="whether to predict reward, default=True")
    parser.add_argument("--ensemble_dim", type=int, default=7, help="ensemble size, default=7")
    parser.add_argument("--topk", type=int, default=5, help="top ensemble to keep when done training, default=5")
    parser.add_argument("--hidden_dim", type=int, default=200, help="neural network hidden dims, default=200")
    parser.add_argument("--num_hidden", type=int, default=3, help="number of hidden layers, default=3")
    parser.add_argument("--activation", type=str, default="silu", help="neural network activation, default=silu")
    parser.add_argument("--residual", type=bool_, default=False, help="whether to predict observation residual, default=False")
    parser.add_argument("--min_std", type=float, default=0.04, help="minimum prediction std, default=0.04")
    parser.add_argument("--max_std", type=float, default=1.6, help="maximum prediction std, default=1.6")
    parser.add_argument("--decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001]")
    # training args
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size, default=256")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate, default=3e-4")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    parser.add_argument("--bootstrap", type=bool_, default=True, 
        help="whether to use different minibatch ordering for each ensemble member, default=True")
    parser.add_argument("--epochs", type=int, default=100, help="number of reward training epochs, default=10")
    parser.add_argument("--max_epochs_since_update", type=int, default=5, help="early stopping condition, default=5")
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--cp_intermediate", type=bool, default=False, help="whether to save intermediate checkpoints, default=False")
    parser.add_argument("--save", type=bool_, default=True)
    arglist = vars(parser.parse_args())
    
    if arglist["yml_path"] != "none":
        print("loaded base config:", arglist["yml_path"])
        with open(arglist["yml_path"], "r") as f:
            yml_args = yaml.safe_load(f)
        arglist.update(yml_args)
    return arglist

class DummyAgent(nn.Module):
    """ Dummy agent wrapper for dynamics training """
    def __init__(self, dynamics, device, lr=1e-3):
        super().__init__()
        self.dynamics = dynamics
        self.device = device

        self.optimizers = {
            "dynamics": torch.optim.Adam(dynamics.parameters(), lr=lr),
        }

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training dynamics offline with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load data
    filepath = os.path.join(arglist["data_path"], arglist["data_name"] + ".p")
    data, _, _, _, _ = load_d4rl_transitions(
        arglist["data_name"], filepath, arglist["num_samples"], 
        shuffle=True, norm_obs=arglist["norm_obs"], norm_rwd=arglist["norm_rwd"]
    )
    
    # init model
    obs_dim = data["obs"].shape[-1]
    act_dim = data["act"].shape[-1]
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        arglist["pred_rwd"],
        arglist["ensemble_dim"],
        arglist["topk"],
        arglist["hidden_dim"],
        arglist["num_hidden"],
        arglist["activation"],
        arglist["decay"],
        residual=arglist["residual"],
        termination_fn=None,
        min_std=arglist["min_std"],
        max_std=arglist["max_std"],
        device=device
    )
    agent = DummyAgent(dynamics, device)
    agent.to(device)
    print(agent)
    
    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["data_name"], arglist["cp_path"])
        agent, cp_history = load_checkpoint(cp_path, agent, device)
    
    # init save callback
    callback = None
    if arglist["save"]:
        plot_keys = ["loss", "mae"]
        save_path = os.path.join(arglist["exp_path"], arglist["data_name"])
        callback = SaveCallback(arglist, save_path, plot_keys, cp_history)
    
    # pack data
    buffer = ReplayBuffer(obs_dim, act_dim, len(data["obs"]))
    buffer.push_batch(data["obs"], data["act"], data["rwd"], data["next_obs"], data["done"])
    data = buffer.sample(buffer.size)
    
    # training loop
    logger = train_ensemble(
        data,
        agent, 
        agent.optimizers["dynamics"],
        arglist["eval_ratio"], 
        arglist["batch_size"], 
        arglist["epochs"], 
        bootstrap=arglist["bootstrap"],
        grad_clip=arglist["grad_clip"], 
        update_elites=True,
        max_epoch_since_update=arglist["max_epochs_since_update"],
        callback=callback, 
        debug=True
    )

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.save_history(pd.DataFrame(logger.history))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
