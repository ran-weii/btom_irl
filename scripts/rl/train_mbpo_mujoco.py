import argparse
import os
import glob
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.agents.dynamics import EnsembleDynamics
from src.agents.mbpo import MBPO
from src.env.gym_wrapper import get_termination_fn
from src.algo.logging_utils import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../../exp/mbpo")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # algo args
    parser.add_argument("--ensemble_dim", type=int, default=7, help="ensemble size, default=7")
    parser.add_argument("--topk", type=int, default=5, help="top k models to perform rollout, default=5")
    parser.add_argument("--hidden_dim", type=int, default=200, help="neural network hidden dims, default=200")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.99")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    parser.add_argument("--clip_lv", type=bool_, default=True, help="whether to clip observation variance, default=True")
    parser.add_argument("--residual", type=bool_, default=False, help="whether to predict observation residual, default=False")
    parser.add_argument("--rwd_clip_max", type=float, default=10., help="clip reward max value, default=10.")
    parser.add_argument("--norm_obs", type=bool_, default=True, help="whether to normalize observation, default=True")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size, default=256")
    parser.add_argument("--rollout_batch_size", type=int, default=50000, help="model rollout batch size, default=50000")
    parser.add_argument("--rollout_min_steps", type=int, default=1, help="min dynamics rollout steps, default=1")
    parser.add_argument("--rollout_max_steps", type=int, default=10, help="max dynamics rollout steps, default=10")
    parser.add_argument("--rollout_min_epoch", type=int, default=20, help="epoch to start increasing rollout steps, default=20")
    parser.add_argument("--rollout_max_epoch", type=int, default=100, help="epoch to stop increasing rollout steps, default=100")
    parser.add_argument("--model_retain_epochs", type=int, default=1, help="number of epochs to retain model samples, default=1")
    parser.add_argument("--real_ratio", type=float, default=0.05, help="ratio of real samples for policy training, default=0.05")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="ratio of real samples for model evaluation, default=0.2")
    parser.add_argument("--m_steps", type=int, default=50, help="model training steps per update, default=50")
    parser.add_argument("--a_steps", type=int, default=50, help="policy training steps per update, default=50")
    parser.add_argument("--lr_a", type=float, default=0.001, help="actor learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--lr_m", type=float, default=0.001, help="model learning rate, default=0.001")
    parser.add_argument("--decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.0001]")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v4", help="environment name, default=Hopper-v4")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=500") 
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--update_model_every", type=int, default=250)
    parser.add_argument("--update_policy_every", type=int, default=50)
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--num_eval_eps", type=int, default=5, help="number of evaluation episodes, default=5")
    parser.add_argument("--eval_deterministic", type=bool_, default=True, help="whether to evaluate deterministically, default=True")
    parser.add_argument("--verbose", type=int, default=10, help="verbose frequency, default=10")
    parser.add_argument("--render", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"training mbpo with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    render_mode = "human" if arglist["render"] else None
    env = gym.make(
        arglist["env_name"], 
        render_mode=render_mode
    )
    env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]
    
    # init agent
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.low.shape[0]
    act_lim = torch.from_numpy(env.action_space.high).to(torch.float32)
    termination_fn = get_termination_fn(arglist["env_name"])
    
    reward = EnsembleDynamics(
        obs_dim,
        act_dim,
        1,
        arglist["ensemble_dim"],
        arglist["topk"],
        arglist["hidden_dim"],
        arglist["num_hidden"],
        arglist["activation"],
        arglist["decay"],
        clip_lv=arglist["clip_lv"],
        residual=False,
        termination_fn=None,
        max_mu=arglist["rwd_clip_max"],
        device=device
    )
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        obs_dim,
        arglist["ensemble_dim"],
        arglist["topk"],
        arglist["hidden_dim"],
        arglist["num_hidden"],
        arglist["activation"],
        arglist["decay"],
        clip_lv=arglist["clip_lv"],
        residual=arglist["residual"],
        termination_fn=termination_fn,
        device=device
    )
    agent = MBPO(
        reward,
        dynamics,
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
        norm_obs=arglist["norm_obs"], 
        buffer_size=arglist["buffer_size"], 
        batch_size=arglist["batch_size"], 
        rollout_batch_size=arglist["rollout_batch_size"], 
        rollout_min_steps=arglist["rollout_min_steps"], 
        rollout_max_steps=arglist["rollout_max_steps"], 
        rollout_min_epoch=arglist["rollout_min_epoch"], 
        rollout_max_epoch=arglist["rollout_max_epoch"], 
        model_retain_epochs=arglist["model_retain_epochs"],
        real_ratio=arglist["real_ratio"], 
        eval_ratio=arglist["eval_ratio"], 
        m_steps=arglist["m_steps"], 
        a_steps=arglist["a_steps"], 
        lr_a=arglist["lr_a"], 
        lr_c=arglist["lr_c"], 
        lr_m=arglist["lr_m"], 
        grad_clip=arglist["grad_clip"], 
        device=device,
    )
    agent.to(device)
    plot_keys = agent.plot_keys
    
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
    eval_env = gym.make(
        arglist["env_name"], 
        render_mode=render_mode
    )
    eval_env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]

    logger = agent.train_policy(
        env, 
        eval_env, 
        arglist["max_steps"], 
        arglist["epochs"], 
        arglist["steps_per_epoch"],
        arglist["update_after"], 
        arglist["update_model_every"], 
        arglist["update_policy_every"], 
        rwd_fn=None, 
        num_eval_eps=arglist["num_eval_eps"], 
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
