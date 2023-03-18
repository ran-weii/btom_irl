import argparse
import os
import glob
import pickle
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from src.algo.offline_irl import OfflineIRL
from src.agents.rl_utils import parse_stacked_trajectories
from src.env.gym_wrapper import GymEnv
from src.algo.logging_utils import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp/offline_irl")
    parser.add_argument("--data_path", type=str, default="../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--dynamics_path", type=str, default="../exp/dynamics/02-08-2023 20-19-31", 
        help="pretrained dynamics path, default=none")
    # algo args
    parser.add_argument("--ensemble_dim", type=int, default=5, help="ensemble size, default=5")
    parser.add_argument("--hidden_dim", type=int, default=128, help="neural network hidden dims, default=128")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=1.")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--clip_lv", type=bool_, default=False, help="whether to clip observation variance, default=False")
    # training args
    parser.add_argument("--rollout_batch_size", type=int, default=10000, help="model rollout batch size, default=10000")
    parser.add_argument("--rollout_steps", type=int, default=100, help="dynamics rollout steps, default=100")
    parser.add_argument("--buffer_size", type=int, default=1e5, help="replay buffer size, default=1e5")
    parser.add_argument("--d_batch_size", type=int, default=10, help="reward training batch size, default=10")
    parser.add_argument("--a_batch_size", type=int, default=200, help="agent training batch size, default=200")
    parser.add_argument("--d_steps", type=int, default=50, help="reward training steps per update, default=30")
    parser.add_argument("--a_steps", type=int, default=50, help="agent training steps per update, default=30")
    parser.add_argument("--lr_d", type=float, default=0.0003, help="reward learning rate, default=0.001")
    parser.add_argument("--lr_a", type=float, default=0.001, help="agent learning rate, default=0.001")
    parser.add_argument("--pess_penalty", type=float, default=0., help="pessimistic penalty, default=0.")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=1e-5")
    parser.add_argument("--grad_clip", type=float, default=100., help="gradient clipping, default=100.")
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of reward training epochs, default=10")
    parser.add_argument("--rl_epochs", type=int, default=5, help="number of rl training epochs, default=5")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=500")
    parser.add_argument("--truncate", type=bool_, default=True, help="whether to truncate episode when unhealthy, default=True")
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--verbose", type=bool_, default=True)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training dynamics offline with settings: {arglist}")

    # load data
    filename = os.path.join(arglist["data_path"], arglist["filename"])
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    
    # unpack dataset
    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"]
    next_obs = dataset["next_observations"]
    terminated = dataset["terminals"]
    timeout = dataset["timeouts"]

    # load dynamics and stats
    assert arglist["dynamics_path"] != "none"
    with open(os.path.join(arglist["dynamics_path"], "norm_stats.npy"), "rb") as f:
        obs_mean = np.load(f)
        obs_std = np.load(f)

    dynamics_state_dict = torch.load(os.path.join(arglist["dynamics_path"], "model.pt"), map_location="cpu")
    print(f"dynamics loaded from: {arglist['dynamics_path']}")

    # normalize data
    obs_norm = (obs - obs_mean) / obs_std
    next_obs_norm = (next_obs - obs_mean) / obs_std

    pad_dataset = parse_stacked_trajectories(
        obs_norm, act, rwd, next_obs_norm, terminated, timeout, max_eps=50
    )
    pad_dataset = [d for d in pad_dataset if sum(d["done"]) == 0]

    # init model
    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]
    act_lim = torch.ones(act_dim)
    agent = OfflineIRL(
        obs_dim, act_dim, act_lim, 
        arglist["ensemble_dim"], arglist["hidden_dim"], arglist["num_hidden"], arglist["activation"],
        gamma=arglist["gamma"], beta=arglist["beta"], polyak=arglist["polyak"], clip_lv=arglist["clip_lv"], 
        rollout_steps=arglist["rollout_steps"], buffer_size=arglist["buffer_size"], d_batch_size=arglist["d_batch_size"], 
        a_batch_size=arglist["a_batch_size"], rollout_batch_size=arglist["rollout_batch_size"], 
        d_steps=arglist["d_steps"], a_steps=arglist["a_steps"], lr_d=arglist["lr_d"], lr_a=arglist["lr_a"], 
        decay=arglist["decay"], grad_clip=arglist["grad_clip"], pess_penalty=arglist["pess_penalty"],
    )
    plot_keys = agent.plot_keys
    
    agent.load_state_dict(dynamics_state_dict["model_state_dict"], strict=False)
    agent.fill_real_buffer(pad_dataset)

    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["cp_path"])
        
        # load state dict
        cp_model_path = glob.glob(os.path.join(cp_path, "models/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1], map_location="cpu")
        agent.load_state_dict(state_dict["model_state_dict"], strict=False)
        for optimizer_name, optimizer_state_dict in state_dict["optimizer_state_dict"].items():
            agent.optimizers[optimizer_name].load_state_dict(optimizer_state_dict)

        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        print(f"loaded checkpoint from {cp_path}\n")
    
    print(agent)
    
    # init save callback
    callback = None
    if arglist["save"]:
        callback = SaveCallback(arglist, plot_keys, cp_history=cp_history)
    
    # training loop
    eval_env = GymEnv(
        "Hopper-v4", 
        obs_mean=obs_mean,
        obs_std=obs_std,
        terminate_when_unhealthy=arglist["truncate"]
    )
    logger = agent.train(
        eval_env, arglist["max_steps"], arglist["epochs"], arglist["rl_epochs"], 
        arglist["steps_per_epoch"], arglist["update_after"], arglist["update_every"],
        callback=callback, verbose=arglist["verbose"]
    )

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.save_history(pd.DataFrame(logger.history))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
