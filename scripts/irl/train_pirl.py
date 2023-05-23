import argparse
import os
import yaml
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from src.utils.data import load_d4rl_transitions, parse_d4rl_stacked_trajectories
from src.algo.reward import Reward
from src.agents.dynamics import EnsembleDynamics, train_ensemble, remove_reward_head
from src.algo.pirl import PIRL
from src.env.gym_wrapper import GymEnv, get_termination_fn
from src.utils.logger import SaveCallback, load_checkpoint

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", type=str, default="pirl")
    parser.add_argument("--base_yml_path", type=str, default="../../config/irl/pirl/base.yml")
    parser.add_argument("--yml_path", type=str, default="../../config/irl/pirl/hopper-medium-expert-v2.yml")
    parser.add_argument("--exp_path", type=str, default="../../exp/mujoco/irl")
    parser.add_argument("--data_path", type=str, default="../../data/d4rl/")
    parser.add_argument("--expert_data_name", type=str, default="hopper-expert-v2")
    parser.add_argument("--transition_data_name", type=str, default="hopper-medium-expert-v2")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--dynamics_path", type=str, default="none", 
        help="pretrained dynamics path, default=none")
    # data args
    parser.add_argument("--num_traj", type=int, default=10, help="number of expert trajectories, default=10")
    parser.add_argument("--num_samples", type=int, default=2000000, help="number of transition samples, default=2e6")
    parser.add_argument("--norm_obs", type=bool_, default=True, help="normalize observatins, default=True")
    parser.add_argument("--norm_rwd", type=bool_, default=False, help="normalize reward, default=False")
    # reward args
    parser.add_argument("--state_only", type=bool_, default=False, help="whether to use state only reward, default=False")
    parser.add_argument("--rwd_clip_max", type=float, default=10., help="clip reward max value, default=10.")
    parser.add_argument("--d_decay", type=float, default=1e-3, help="reward weight decay, default=1e-3")
    parser.add_argument("--grad_penalty", type=float, default=1., help="gradient penalty, default=1.")
    parser.add_argument("--grad_target", type=float, default=1., help="gradient target, default=1.")
    parser.add_argument("--rwd_rollout_batch_size", type=int, default=64, help="reward rollout batch size, default=64")
    parser.add_argument("--rwd_rollout_steps", type=int, default=100, help="reward rollout steps, default=100")
    parser.add_argument("--rwd_update_method", type=str, choices=["traj", "marginal"], default="traj")
    # dynamics args
    parser.add_argument("--ensemble_dim", type=int, default=7, help="ensemble size, default=7")
    parser.add_argument("--topk", type=int, default=5, help="top k models to perform rollout, default=5")
    parser.add_argument("--m_hidden_dim", type=int, default=200, help="dynamics neural network hidden dims, default=200")
    parser.add_argument("--m_num_hidden", type=int, default=3, help="dynamics number of hidden layers, default=3")
    parser.add_argument("--m_activation", type=str, default="silu", help="dynamics neural network activation, default=silu")
    parser.add_argument("--residual", type=bool_, default=True, help="whether to predict observation residual, default=True")
    parser.add_argument("--min_std", type=float, default=0.04, help="dynamics minimum prediction std, default=0.04")
    parser.add_argument("--max_std", type=float, default=1.6, help="dynamics maximum prediction std, default=1.6")
    parser.add_argument("--m_decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001]")
    parser.add_argument("--lam", type=float, default=1., help="mopo penalty, default=1.")
    parser.add_argument("--lam_target", type=float, default=1., help="mopo penalty target, default=1.5")
    parser.add_argument("--tune_lam", type=bool_, default=True, help="whether to tune mopo penalty, default=True")
    # policy args
    parser.add_argument("--a_hidden_dim", type=int, default=256, help="policy neural network hidden dims, default=256")
    parser.add_argument("--a_num_hidden", type=int, default=2, help="policy number of hidden layers, default=2")
    parser.add_argument("--a_activation", type=str, default="relu", help="policy neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.99")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--min_beta", type=float, default=0.001, help="minimum softmax temperature, default=0.001")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    # training args
    parser.add_argument("--buffer_size", type=int, default=2e6, help="replay buffer size, default=2e6")
    parser.add_argument("--batch_size", type=int, default=256, help="policy training batch size, default=256")
    parser.add_argument("--rollout_batch_size", type=int, default=50000, help="model rollout batch size, default=50000")
    parser.add_argument("--rollout_deterministic", type=bool_, default=False, help="whether to rollout deterministically, default=False")
    parser.add_argument("--rollout_min_steps", type=int, default=5, help="min dynamics rollout steps, default=5")
    parser.add_argument("--rollout_max_steps", type=int, default=5, help="max dynamics rollout steps, default=5")
    parser.add_argument("--rollout_min_epoch", type=int, default=20, help="epoch to start increasing rollout steps, default=20")
    parser.add_argument("--rollout_max_epoch", type=int, default=100, help="epoch to stop increasing rollout steps, default=100")
    parser.add_argument("--model_retain_epochs", type=int, default=5, help="number of epochs to retain model samples, default=5")
    parser.add_argument("--real_ratio", type=float, default=0.5, help="ratio of real samples for policy training, default=0.5")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="ratio of real samples for model evaluation, default=0.2")
    parser.add_argument("--a_steps", type=int, default=1, help="policy training steps per update, default=1")
    parser.add_argument("--d_steps", type=int, default=50, help="reward training steps per update, default=30")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="actor learning rate, default=3e-4")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="critic learning rate, default=3e-4")
    parser.add_argument("--lr_lam", type=float, default=0.01, help="penalty learning rate, default=0.01")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="reward learning rate, default=1e-4")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v2", help="environment name, default=Hopper-v2")
    parser.add_argument("--pretrain_steps", type=int, default=50, help="number of dynamics and reward pretraining steps, default=50")
    parser.add_argument("--num_pretrain_samples", type=int, default=100000, help="number of dynamics and reward pretraining samples, default=1e5")
    parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs, default=2000")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--sample_model_every", type=int, default=250)
    parser.add_argument("--update_model_every", type=int, default=1000)
    parser.add_argument("--update_policy_every", type=int, default=1)
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
    print(f"training {arglist['algo']} with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load data
    transition_data_path = os.path.join(arglist["data_path"], arglist["transition_data_name"] + ".p")
    data, obs_mean, obs_std, _, _ = load_d4rl_transitions(
        arglist["transition_data_name"], transition_data_path, arglist["num_samples"], shuffle=True,
        norm_obs=arglist["norm_obs"], norm_rwd=arglist["norm_rwd"]
    )
    expert_data_path = os.path.join(arglist["data_path"], arglist["expert_data_name"] + ".p")
    expert_dataset = parse_d4rl_stacked_trajectories(
        arglist["expert_data_name"], expert_data_path, arglist["num_traj"], skip_terminated=True,
        obs_mean=obs_mean, obs_std=obs_std
    )
    
    # init model
    obs_dim = data["obs"].shape[-1]
    act_dim = data["act"].shape[-1]
    act_lim = torch.ones(act_dim)
    termination_fn = get_termination_fn(
        arglist["env_name"], 
        obs_mean=obs_mean, 
        obs_variance=obs_std**2
    )
    
    reward = Reward(
        obs_dim,
        act_dim,
        arglist["a_hidden_dim"], 
        arglist["a_num_hidden"], 
        arglist["a_activation"],
        state_only=arglist["state_only"],
        clip_max=arglist["rwd_clip_max"],
        decay=arglist["d_decay"],
        grad_penalty=arglist["grad_penalty"],
        grad_target=arglist["grad_target"],
        device=device
    )
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        pred_rwd=False,
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
    agent = PIRL(
        reward,
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
        rwd_rollout_batch_size=arglist["rwd_rollout_batch_size"],
        rwd_rollout_steps=arglist["rwd_rollout_steps"],
        rwd_update_method=arglist["rwd_update_method"],
        lam=arglist["lam"],
        lam_target=arglist["lam_target"],
        tune_lam=arglist["tune_lam"],
        buffer_size=arglist["buffer_size"], 
        batch_size=arglist["batch_size"], 
        rollout_batch_size=arglist["rollout_batch_size"], 
        rollout_deterministic=arglist["rollout_deterministic"], 
        rollout_min_steps=arglist["rollout_min_steps"], 
        rollout_max_steps=arglist["rollout_max_steps"], 
        rollout_min_epoch=arglist["rollout_min_epoch"], 
        rollout_max_epoch=arglist["rollout_max_epoch"], 
        model_retain_epochs=arglist["model_retain_epochs"],
        real_ratio=arglist["real_ratio"], 
        eval_ratio=arglist["eval_ratio"],
        a_steps=arglist["a_steps"], 
        d_steps=arglist["d_steps"], 
        lr_a=arglist["lr_a"], 
        lr_c=arglist["lr_c"], 
        lr_lam=arglist["lr_lam"], 
        lr_d=arglist["lr_d"], 
        grad_clip=arglist["grad_clip"], 
        device=device,
    )
    agent.to(device)
    plot_keys = agent.plot_keys
    
    if arglist["dynamics_path"] != "none":
        dynamics_state_dict = torch.load(os.path.join(arglist["dynamics_path"], "models", "model.pt"), map_location=device)
        dynamics_state_dict = remove_reward_head(dynamics_state_dict, obs_dim, arglist["m_num_hidden"])
        agent.load_state_dict(dynamics_state_dict["model_state_dict"], strict=False)
        print(f"dynamics loaded from: {arglist['dynamics_path']}")

    agent.real_buffer.push_batch(
        data["obs"], 
        data["act"], 
        data["rwd"], 
        data["next_obs"], 
        data["done"]
    )
    agent.fill_expert_buffer(expert_dataset)

    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["transition_data_name"], arglist["cp_path"])
        agent, cp_history = load_checkpoint(cp_path, agent, device)
    
    print(agent)
    print(f"transition buffer size: {agent.real_buffer.size}")
    print(f"expert buffer size: {agent.expert_buffer.size}")
    
    # init save callback
    callback = None
    if arglist["save"]:
        save_path = os.path.join(arglist["exp_path"], arglist["algo"], arglist["transition_data_name"])
        callback = SaveCallback(arglist, save_path, plot_keys, cp_history)
    
    # training loop
    render_mode = "human" if arglist["render"] else None
    eval_env = GymEnv(
        arglist["env_name"], 
        obs_mean=obs_mean, 
        obs_variance=obs_std**2,
        render_mode=render_mode,
    )
    eval_env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]
    
    print("\npretrain dynamics:", arglist["pretrain_steps"] > 0)
    if arglist["dynamics_path"] == "none" or arglist["pretrain_steps"] > 0:
        data = agent.real_buffer.sample(arglist["num_pretrain_samples"])
        dynamics_pretrain_optimizer = torch.optim.Adam(agent.dynamics.parameters(), lr=arglist["lr_m"])
        train_ensemble(
            data, 
            agent,
            optimizer=dynamics_pretrain_optimizer,
            eval_ratio=arglist["eval_ratio"],
            batch_size=arglist["batch_size"],
            epochs=arglist["pretrain_steps"],
            bootstrap=True,
            grad_clip=arglist["grad_clip"],
            update_elites=True,
            max_epoch_since_update=10,
            debug=True
        )

    logger = agent.train(
        eval_env,  
        arglist["epochs"], 
        arglist["steps_per_epoch"], 
        arglist["sample_model_every"],
        arglist["update_model_every"],
        eval_steps=arglist["eval_steps"],
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
