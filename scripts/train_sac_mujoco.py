import argparse
import os
import glob
import json
import datetime
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.agents.sac import SAC
from src.agents.rl_utils import train

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # algo args
    parser.add_argument("--hidden_dim", type=int, default=128, help="neural network hidden dims, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.1")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observations for agent and algo, default=False")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=200, help="training batch size, default=200")
    parser.add_argument("--steps", type=int, default=50, help="training steps per update, default=30")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=1e-5")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=500")
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--verbose", type=bool_, default=True)
    parser.add_argument("--render", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

def plot_history(df_history, plot_keys, plot_std=True):
    """ Plot learning history
    
    Args:
        df_history (pd.dataframe): learning history dataframe with a binary train column.
        plot_keys (list): list of column names to be plotted.
        plot_std (bool, optional): whether to plot std shade. Default=True

    Returns:
        fig (plt.figure)
        ax (plt.axes)
    """
    num_cols = len(plot_keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_history["epoch"], df_history[plot_keys[i]])
        if plot_std:
            std = df_history[plot_keys[i].replace("_avg", "_std")]
            ax[i].fill_between(
                df_history["epoch"],
                df_history[plot_keys[i]] - std,
                df_history[plot_keys[i]] + std,
                alpha=0.4
            )

        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(plot_keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax

class SaveCallback:
    def __init__(self, arglist, plot_keys, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist["exp_path"], "sac")
        save_path = os.path.join(exp_path, date_time)
        model_path = os.path.join(save_path, "models") # used to save model checkpoint
        if not os.path.exists(arglist["exp_path"]):
            os.mkdir(arglist["exp_path"])
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(arglist, f)

        self.save_path = save_path
        self.model_path = model_path
        self.plot_keys = plot_keys
        self.cp_history = cp_history
        self.cp_every = arglist["cp_every"]
        self.iter = 0

    def __call__(self, model, logger):
        self.iter += 1
        if self.iter % self.cp_every != 0:
            return
        
        # save history
        df_history = pd.DataFrame(logger.history)
        self.save_history(df_history)
        
        # save model
        self.save_checkpoint(model, os.path.join(self.model_path, f"model_{self.iter}.pt"))
    
    def save_history(self, df_history):
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, self.plot_keys)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)

        plt.clf()
        plt.close()

    def save_checkpoint(self, model, path=None):
        if path is None:
            path = os.path.join(self.save_path, "model.pt")

        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        actor_optimizer_state_dict = {
            k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in model.actor_optimizer.state_dict().items()
        }
        critic_optimizer_state_dict = {
            k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in model.critic_optimizer.state_dict().items()
        }
        
        torch.save({
            "model_state_dict": model_state_dict,
            "actor_optimizer_state_dict": actor_optimizer_state_dict,
            "critic_optimizer_state_dict": critic_optimizer_state_dict,
        }, path)
        print(f"\ncheckpoint saved at: {path}\n")

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training sac with settings: {arglist}")
    
    render_mode = "human" if arglist["render"] else None
    env = gym.make("Hopper-v4", terminate_when_unhealthy=True, render_mode=render_mode)
    
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.low.shape[0]
    act_lim = torch.from_numpy(env.action_space.high).to(torch.float32)
    
    agent = SAC(
        obs_dim, act_dim, act_lim, arglist["hidden_dim"], arglist["num_hidden"], arglist["activation"],
        gamma=arglist["gamma"], beta=arglist["beta"], polyak=arglist["polyak"],
        norm_obs=arglist["norm_obs"], buffer_size=arglist["buffer_size"], batch_size=arglist["batch_size"],
        steps=arglist["steps"], lr=arglist["lr"], decay=arglist["decay"], grad_clip=arglist["grad_clip"], 
    )
    plot_keys = agent.plot_keys
    
    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], "sac", arglist["cp_path"])
        
        # load state dict
        cp_model_path = glob.glob(os.path.join(cp_path, "models/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1], map_location="cpu")
        agent.load_state_dict(state_dict["model_state_dict"], strict=False)
        agent.actor_optimizer.load_state_dict(state_dict["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(state_dict["critic_optimizer_state_dict"])

        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        print(f"loaded checkpoint from {cp_path}\n")
    
    print(agent)
    
    # init save callback
    callback = None
    if arglist["save"]:
        callback = SaveCallback(arglist, plot_keys, cp_history=cp_history)
    
    # training loop
    agent, logger = train(
        env, agent, arglist["epochs"], max_steps=arglist["max_steps"], 
        steps_per_epoch=arglist["steps_per_epoch"], update_after=arglist["update_after"], 
        update_every=arglist["update_every"], verbose=arglist["verbose"], callback=callback, render=arglist["render"]
    )

    if arglist["save"]:
        callback.save_checkpoint(agent)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
