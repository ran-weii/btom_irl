import argparse
import os
import yaml
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.agents.dynamics import EnsembleDynamics
from src.agents.mbpo import MBPO
from src.env.gym_wrapper import get_termination_fn
from src.utils.logger import SaveCallback, load_checkpoint

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", type=str, default="mbpo")
    parser.add_argument("--base_yml_path", type=str, default="../../config/rl/mbpo/base.yml")
    parser.add_argument("--yml_path", type=str, default="../../config/rl/mbpo/hopper.yml")
    parser.add_argument("--exp_path", type=str, default="../../exp/mujoco/rl")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # dynamics args
    parser.add_argument("--ensemble_dim", type=int, default=7, help="ensemble size, default=7")
    parser.add_argument("--topk", type=int, default=5, help="top k models to perform rollout, default=5")
    parser.add_argument("--m_hidden_dim", type=int, default=200, help="dynamics neural network hidden dims, default=200")
    parser.add_argument("--m_num_hidden", type=int, default=2, help="dynamics number of hidden layers, default=2")
    parser.add_argument("--m_activation", type=str, default="relu", help="dynamics neural network activation, default=relu")
    parser.add_argument("--residual", type=bool_, default=False, help="whether to predict observation residual, default=False")
    parser.add_argument("--min_std", type=float, default=1e-5, help="dynamics minimum prediction std, default=1e-5")
    parser.add_argument("--max_std", type=float, default=1.6, help="dynamics maximum prediction std, default=1.6")
    parser.add_argument("--norm_obs", type=bool_, default=True, help="whether to normalize observation, default=True")
    parser.add_argument("--m_decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.0001]")
    # policy args
    parser.add_argument("--a_hidden_dim", type=int, default=200, help="policy neural network hidden dims, default=200")
    parser.add_argument("--a_num_hidden", type=int, default=2, help="policy number of hidden layers, default=2")
    parser.add_argument("--a_activation", type=str, default="relu", help="policy neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.99")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--min_beta", type=float, default=0.001, help="minimum softmax temperature, default=0.001")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size, default=256")
    parser.add_argument("--rollout_batch_size", type=int, default=50000, help="model rollout batch size, default=50000")
    parser.add_argument("--rollout_deterministic", type=bool_, default=False, help="whether to rollout deterministically, default=False")
    parser.add_argument("--rollout_min_steps", type=int, default=1, help="min dynamics rollout steps, default=1")
    parser.add_argument("--rollout_max_steps", type=int, default=10, help="max dynamics rollout steps, default=10")
    parser.add_argument("--rollout_min_epoch", type=int, default=20, help="epoch to start increasing rollout steps, default=20")
    parser.add_argument("--rollout_max_epoch", type=int, default=100, help="epoch to stop increasing rollout steps, default=100")
    parser.add_argument("--model_retain_epochs", type=int, default=1, help="number of epochs to retain model samples, default=1")
    parser.add_argument("--model_train_samples", type=int, default=100000, help="maximum number of samples for model training, default=1e5")
    parser.add_argument("--real_ratio", type=float, default=0.05, help="ratio of real samples for policy training, default=0.05")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="ratio of real samples for model evaluation, default=0.2")
    parser.add_argument("--m_steps", type=int, default=50, help="model training steps per update, default=50")
    parser.add_argument("--a_steps", type=int, default=50, help="policy training steps per update, default=50")
    parser.add_argument("--lr_a", type=float, default=0.001, help="actor learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--lr_m", type=float, default=0.001, help="model learning rate, default=0.001")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v4", help="environment name, default=Hopper-v4")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=500") 
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--update_after", type=int, default=2000)
    parser.add_argument("--update_model_every", type=int, default=250)
    parser.add_argument("--update_policy_every", type=int, default=50)
    parser.add_argument("--num_eval_eps", type=int, default=5, help="number of evaluation episodes, default=5")
    parser.add_argument("--eval_steps", type=int, default=1000, help="number of evaluation steps, default=1000")
    parser.add_argument("--eval_deterministic", type=bool_, default=True, help="whether to evaluate deterministically, default=True")
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--cp_intermediate", type=bool, default=False, help="whether to save intermediate checkpoints, default=False")
    parser.add_argument("--verbose", type=int, default=10, help="verbose frequency, default=10")
    parser.add_argument("--render", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = vars(parser.parse_args())

    if arglist["base_yml_path"] != "none":
        print("loaded base config:", arglist["base_yml_path"])
        with open(arglist["base_yml_path"], "r") as f:
            base_yml_args = yaml.safe_load(f)
        arglist.update(base_yml_args)

    if arglist["yml_path"] != "none":
        print("loaded task config:", arglist["yml_path"])
        with open(arglist["yml_path"], "r") as f:
            yml_args = yaml.safe_load(f)
        arglist.update(yml_args)
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
    
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        pred_rwd=True,
        ensemble_dim=arglist["ensemble_dim"],
        topk=arglist["topk"],
        hidden_dim=arglist["m_hidden_dim"],
        num_hidden=arglist["m_num_hidden"],
        activation=arglist["m_activation"],
        decay=arglist["m_decay"],
        residual=arglist["residual"],
        termination_fn=termination_fn,
        min_std=arglist["min_std"],
        max_std=arglist["max_std"],
        device=device
    )
    agent = MBPO(
        dynamics,
        obs_dim, 
        act_dim, 
        act_lim, 
        arglist["a_hidden_dim"], 
        arglist["a_num_hidden"], 
        arglist["a_activation"],
        gamma=arglist["gamma"], 
        beta=arglist["beta"], 
        min_beta=arglist["min_beta"],
        polyak=arglist["polyak"], 
        tune_beta=arglist["tune_beta"], 
        norm_obs=arglist["norm_obs"], 
        buffer_size=arglist["buffer_size"], 
        batch_size=arglist["batch_size"], 
        rollout_batch_size=arglist["rollout_batch_size"], 
        rollout_deterministic=arglist["rollout_deterministic"], 
        rollout_min_steps=arglist["rollout_min_steps"], 
        rollout_max_steps=arglist["rollout_max_steps"], 
        rollout_min_epoch=arglist["rollout_min_epoch"], 
        rollout_max_epoch=arglist["rollout_max_epoch"], 
        model_retain_epochs=arglist["model_retain_epochs"],
        model_train_samples=arglist["model_train_samples"],
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
        cp_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["env_name"], arglist["cp_path"])
        agent, cp_history = load_checkpoint(cp_path, agent, device)
    
    print(agent)
    
    # init save callback
    callback = None
    if arglist["save"]:
        save_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["env_name"])
        callback = SaveCallback(arglist, save_path, plot_keys, cp_history)
    
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
