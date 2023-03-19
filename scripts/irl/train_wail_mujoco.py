import argparse
import os
import glob
import pickle
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import torch 

from src.algo.wail import WAIL
from src.utils.data import parse_stacked_trajectories
from src.utils.logging import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../../exp/wail")
    parser.add_argument("--data_path", type=str, default="../../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--num_traj", type=int, default=50, help="number of training trajectories, default=50")
    # algo args
    parser.add_argument("--hidden_dim", type=int, default=128, help="neural network hidden dims, default=128")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=1.")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    parser.add_argument("--rwd_clip_max", type=float, default=10., help="clip reward max value, default=10.")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e5, help="replay buffer size, default=1e5")
    parser.add_argument("--batch_size", type=int, default=10, help="reward training batch size, default=10")
    parser.add_argument("--real_ratio", type=float, default=0.5, help="ratio of real samples for policy training, default=0.05")
    parser.add_argument("--d_steps", type=int, default=50, help="reward training steps per update, default=30")
    parser.add_argument("--a_steps", type=int, default=50, help="agent training steps per update, default=30")
    parser.add_argument("--lr_d", type=float, default=0.0003, help="reward learning rate, default=0.001")
    parser.add_argument("--lr_a", type=float, default=0.001, help="agent learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="reward weight decay, default=1e-5")
    parser.add_argument("--grad_clip", type=float, default=100., help="gradient clipping, default=100.")
    parser.add_argument("--grad_penalty", type=float, default=1., help="gradient penalty, default=1.")
    parser.add_argument("--grad_target", type=float, default=1., help="gradient target, default=1.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v4", help="environment name, default=Hopper-v4")
    parser.add_argument("--epochs", type=int, default=100, help="number of reward training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=500")
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=1000, help="number of evaluation steps, default=1000")
    parser.add_argument("--num_eval_eps", type=int, default=5, help="number of evaluation episodes, default=5")
    parser.add_argument("--eval_deterministic", type=bool_, default=True, help="whether to evaluate deterministically, default=True")
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--verbose", type=int, default=10, help="verbose frequency, default=10")
    parser.add_argument("--render", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training wail with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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
    
    pad_dataset = parse_stacked_trajectories(
        obs, act, rwd, next_obs, terminated, timeout, max_eps=arglist["num_traj"]
    )
    
    # init agent
    obs_dim = pad_dataset[0]["obs"].shape[-1]
    act_dim = pad_dataset[0]["act"].shape[-1]
    act_lim = torch.ones(act_dim)

    agent = WAIL(
        obs_dim, 
        act_dim, 
        act_lim, 
        arglist["hidden_dim"], 
        arglist["num_hidden"], 
        arglist["activation"],
        gamma=arglist["gamma"], 
        beta=arglist["beta"], 
        polyak=arglist["polyak"],
        tune_beta=arglist["tune_beta"],
        rwd_clip_max=arglist["rwd_clip_max"],
        buffer_size=arglist["buffer_size"],
        batch_size=arglist["batch_size"], 
        real_ratio=arglist["real_ratio"],
        d_steps=arglist["d_steps"], 
        a_steps=arglist["a_steps"], 
        lr_a=arglist["lr_a"], 
        lr_c=arglist["lr_c"], 
        lr_d=arglist["lr_d"], 
        decay=arglist["decay"], 
        grad_clip=arglist["grad_clip"],
        grad_penalty=arglist["grad_penalty"],
        grad_target=arglist["grad_target"],
        device=device
    )
    agent.to(device)
    plot_keys = agent.plot_keys

    agent.fill_real_buffer(pad_dataset)
    
    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["cp_path"])
        
        # load state dict
        cp_model_path = glob.glob(os.path.join(cp_path, "models/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1], map_location=device)
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
    render_mode = "human" if arglist["render"] else None
    env = gym.make(
        "Hopper-v4", 
        render_mode=render_mode
    )
    env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]
    eval_env = gym.make(
        "Hopper-v4", 
        render_mode=render_mode
    )
    eval_env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]

    logger = agent.train(
        env, 
        eval_env, 
        arglist["max_steps"], 
        arglist["epochs"], 
        arglist["steps_per_epoch"], 
        arglist["update_after"], 
        arglist["update_every"],
        eval_steps=arglist["eval_steps"],
        num_eval_eps=arglist["num_eval_eps"],
        eval_deterministic=arglist["eval_deterministic"],
        callback=callback, 
        verbose=arglist["verbose"]
    )

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.tb_writer.close()

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
