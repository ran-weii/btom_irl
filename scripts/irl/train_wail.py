import argparse
import os
import yaml
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import torch 

from src.algo.reward import Reward
from src.algo.wail import WAIL
from src.utils.data import parse_d4rl_stacked_trajectories
from src.utils.logger import SaveCallback, load_checkpoint

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", type=str, default="wail")
    parser.add_argument("--yml_path", type=str, default="../../config/irl/wail/base.yml")
    parser.add_argument("--exp_path", type=str, default="../../exp/mujoco/irl")
    parser.add_argument("--data_path", type=str, default="../../data/d4rl/")
    parser.add_argument("--data_name", type=str, default="hopper-expert-v2")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # data args
    parser.add_argument("--num_traj", type=int, default=10, help="number of training trajectories, default=10")
    # reward args
    parser.add_argument("--state_only", type=bool_, default=False, help="whether to use state only reward, default=False")
    parser.add_argument("--rwd_clip_max", type=float, default=10., help="clip reward max value, default=10.")
    parser.add_argument("--d_decay", type=float, default=1e-5, help="reward weight decay, default=1e-5")
    parser.add_argument("--grad_penalty", type=float, default=1., help="reward gradient penalty, default=1.")
    parser.add_argument("--grad_target", type=float, default=1., help="rewrd gradient target, default=1.")
    # policy args
    parser.add_argument("--hidden_dim", type=int, default=128, help="neural network hidden dims, default=128")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=1.")
    parser.add_argument("--min_beta", type=float, default=0.001, help="minimum softmax temperature, default=0.001")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=256, help="reward training batch size, default=256")
    parser.add_argument("--real_ratio", type=float, default=0.5, help="ratio of real samples for policy training, default=0.05")
    parser.add_argument("--d_steps", type=int, default=50, help="reward training steps per update, default=30")
    parser.add_argument("--a_steps", type=int, default=50, help="agent training steps per update, default=30")
    parser.add_argument("--lr_d", type=float, default=0.0003, help="reward learning rate, default=0.0003")
    parser.add_argument("--lr_a", type=float, default=0.0003, help="agent learning rate, default=0.0003")
    parser.add_argument("--lr_c", type=float, default=0.0003, help="critic learning rate, default=0.0003")
    parser.add_argument("--grad_clip", type=float, default=100., help="gradient clipping, default=100.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v2", help="environment name, default=Hopper-v4")
    parser.add_argument("--epochs", type=int, default=1000, help="number of reward training epochs, default=1000")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=1000")
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--num_eval_eps", type=int, default=5, help="number of evaluation episodes, default=5")
    parser.add_argument("--eval_steps", type=int, default=1000, help="number of evaluation steps, default=1000")
    parser.add_argument("--eval_deterministic", type=bool_, default=True, help="whether to evaluate deterministically, default=True")
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--cp_intermediate", type=bool, default=False, help="whether to save intermediate checkpoints, default=False")
    parser.add_argument("--verbose", type=int, default=10, help="verbose frequency, default=10")
    parser.add_argument("--render", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = vars(parser.parse_args())

    if arglist["yml_path"] != "none":
        print("loaded base config:", arglist["yml_path"])
        with open(arglist["yml_path"], "r") as f:
            yml_args = yaml.safe_load(f)
        arglist.update(yml_args)
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training wail with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load data
    filepath = os.path.join(arglist["data_path"], arglist["data_name"] + ".p")
    traj_dataset = parse_d4rl_stacked_trajectories(
        arglist["data_name"], filepath, arglist["num_traj"], skip_terminated=True,
    )
    
    # init agent
    obs_dim = traj_dataset[0]["obs"].shape[-1]
    act_dim = traj_dataset[0]["act"].shape[-1]
    act_lim = torch.ones(act_dim)
    
    reward = Reward(
        obs_dim,
        act_dim,
        arglist["hidden_dim"], 
        arglist["num_hidden"], 
        arglist["activation"],
        state_only=arglist["state_only"],
        clip_max=arglist["rwd_clip_max"],
        decay=arglist["d_decay"],
        grad_penalty=arglist["grad_penalty"],
        grad_target=arglist["grad_target"],
        device=device
    )
    agent = WAIL(
        reward,
        obs_dim, 
        act_dim, 
        act_lim, 
        arglist["hidden_dim"], 
        arglist["num_hidden"], 
        arglist["activation"],
        gamma=arglist["gamma"], 
        beta=arglist["beta"], 
        min_beta=arglist["min_beta"],
        polyak=arglist["polyak"],
        tune_beta=arglist["tune_beta"],
        buffer_size=arglist["buffer_size"],
        batch_size=arglist["batch_size"], 
        real_ratio=arglist["real_ratio"],
        d_steps=arglist["d_steps"], 
        a_steps=arglist["a_steps"], 
        lr_a=arglist["lr_a"], 
        lr_c=arglist["lr_c"], 
        lr_d=arglist["lr_d"], 
        grad_clip=arglist["grad_clip"],
        device=device
    )
    agent.to(device)
    plot_keys = agent.plot_keys

    agent.fill_expert_buffer(traj_dataset)
    
    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["data_name"], arglist["cp_path"])
        agent, cp_history = load_checkpoint(cp_path, agent, device)
    
    print(agent)
    
    # init save callback
    callback = None
    if arglist["save"]:
        save_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["data_name"])
        callback = SaveCallback(arglist, save_path, plot_keys, cp_history)
    
    # training loop
    render_mode = "human" if arglist["render"] else None
    env = gym.make(
        arglist["env_name"], 
        render_mode=render_mode
    )
    env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]
    eval_env = gym.make(
        arglist["env_name"], 
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
        num_eval_eps=arglist["num_eval_eps"],
        eval_steps=arglist["eval_steps"],
        eval_deterministic=arglist["eval_deterministic"],
        callback=callback, 
        verbose=arglist["verbose"]
    )

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.save_history(pd.DataFrame(logger.history))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
