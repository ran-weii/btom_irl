import argparse
import os
import glob
import pickle
import mujoco_py
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.agents.dynamics import EnsembleDynamics, train_ensemble
from src.agents.rambo import RAMBO
from src.env.gym_wrapper import GymEnv, get_termination_fn
from src.utils.logging import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../../exp/mujoco/rl")
    parser.add_argument("--data_path", type=str, default="../../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--dynamics_path", type=str, default="", 
        help="pretrained dynamics path, default=none")
    # data args
    parser.add_argument("--num_samples", type=int, default=100000, help="number of training transitions, default=100000")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="normalize observatins, default=False")
    parser.add_argument("--norm_rwd", type=bool_, default=False, help="normalize reward, default=False")
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
    parser.add_argument("--obs_penalty", type=float, default=1., help="transition likelihood penalty, default=1.")
    parser.add_argument("--adv_penalty", type=float, default=3e-4, help="model advantage penalty, default=3e-4")
    parser.add_argument("--norm_advantage", type=bool_, default=True, help="whether to normalize advantage, default=True")
    parser.add_argument("--update_critic_adv", type=bool_, default=False, help="whether to update critic during model training, default=False")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size, default=256")
    parser.add_argument("--rollout_batch_size", type=int, default=50000, help="model rollout batch size, default=50000")
    parser.add_argument("--rollout_min_steps", type=int, default=1, help="min dynamics rollout steps, default=1")
    parser.add_argument("--rollout_max_steps", type=int, default=10, help="max dynamics rollout steps, default=10")
    parser.add_argument("--rollout_min_epoch", type=int, default=20, help="epoch to start increasing rollout steps, default=20")
    parser.add_argument("--rollout_max_epoch", type=int, default=100, help="epoch to stop increasing rollout steps, default=100")
    parser.add_argument("--model_retain_epochs", type=int, default=4, help="number of epochs to retain model samples, default=4")
    parser.add_argument("--real_ratio", type=float, default=0.5, help="ratio of real samples for policy training, default=0.5")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="ratio of real samples for model evaluation, default=0.2")
    parser.add_argument("--m_steps", type=int, default=1000, help="model training steps per update, default=1000")
    parser.add_argument("--a_steps", type=int, default=1, help="policy training steps per update, default=1")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="actor learning rate, default=1e-4")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="critic learning rate, default=3e-4")
    parser.add_argument("--lr_m", type=float, default=3e-4, help="model learning rate, default=3e-4")
    parser.add_argument("--decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.0001]")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    # rollout args
    parser.add_argument("--env_name", type=str, default="Hopper-v4", help="environment name, default=Hopper-v4")
    parser.add_argument("--pretrain_steps", type=int, default=50, help="number of dynamics and reward pretraining steps, default=50")
    parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs, default=2000")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per episode, default=1000")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--sample_model_every", type=int, default=250)
    parser.add_argument("--update_model_every", type=int, default=1000)
    parser.add_argument("--update_policy_every", type=int, default=1)
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
    print(f"training rambo with settings: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load data
    filename = os.path.join(arglist["data_path"], arglist["filename"])
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    
    # unpack dataset
    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"].reshape(-1, 1)
    next_obs = dataset["next_observations"]
    terminated = dataset["terminals"].reshape(-1, 1)
    
    # subsample data
    num_samples = arglist["num_samples"]
    idx = np.arange(len(obs))
    np.random.shuffle(idx)
    idx = idx[:num_samples]

    obs = obs[idx]
    act = act[idx]
    rwd = rwd[idx]
    next_obs = next_obs[idx]
    terminated = terminated[idx]
    
    # normalize data
    obs_mean = 0.
    obs_std = 1.
    if arglist["norm_obs"]:
        obs_mean = obs.mean(0)
        obs_std = obs.std(0)
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std
    
    rwd_mean = 0.
    rwd_std = 1.
    if arglist["norm_rwd"]:
        rwd_mean = rwd.mean(0)
        rwd_std = rwd.std(0)
        rwd = (rwd - rwd_mean) / rwd_std
    
    print("processed data stats")
    print("obs_mean:", obs.mean(0).round(2))
    print("obs_std:", obs.std(0).round(2))
    print("rwd_mean:", rwd.mean(0).round(2))
    print("rwd_std:", rwd.std(0).round(2))

    # init model
    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]
    act_lim = torch.ones(act_dim)
    termination_fn = get_termination_fn(
        arglist["env_name"], 
        obs_mean=obs_mean, 
        obs_variance=obs_std**2
    )

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
    agent = RAMBO(
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
        obs_penalty=arglist["obs_penalty"], 
        adv_penalty=arglist["adv_penalty"], 
        norm_advantage=arglist["norm_advantage"],
        update_critic_adv=arglist["update_critic_adv"],
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

    if arglist["dynamics_path"] != "none":
        dynamics_state_dict = torch.load(os.path.join(arglist["dynamics_path"], "model.pt"), map_location=device)
        agent.load_state_dict(dynamics_state_dict["model_state_dict"], strict=False)
        print(f"dynamics loaded from: {arglist['dynamics_path']}")
    
    agent.real_buffer.push_batch(
        obs, act, rwd, next_obs, terminated
    )
    agent.update_stats()
    
    # load checkpoint
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["env_name"], "rambo", arglist["cp_path"])
        
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
    print(f"real buffer size: {agent.real_buffer.size}")
    
    # init save callback
    callback = None
    if arglist["save"]:
        save_path = os.path.join(arglist["exp_path"], arglist["env_name"], "rambo")
        callback = SaveCallback(arglist, save_path)
    
    # training loop
    render_mode = "human" if arglist["render"] else None
    eval_env = GymEnv(
        arglist["env_name"], 
        obs_mean=obs_mean, 
        obs_variance=obs_std**2,
        rwd_mean=rwd_mean,
        rwd_variance=rwd_std**2, 
        render_mode=render_mode,
    )
    eval_env.np_random = gym.utils.seeding.np_random(arglist["seed"])[0]
    
    print("\npretrain dynamics:", arglist["pretrain_steps"] > 0)
    train_ensemble(
        [obs, act, rwd, next_obs],
        agent,
        arglist["eval_ratio"],
        arglist["batch_size"],
        arglist["pretrain_steps"],
        grad_clip=arglist["grad_clip"],
        train_reward=True,
        update_stats=True,
        update_elites=True,
        max_epoch_since_update=10,
        verbose=1,
    )
    
    logger = agent.train_policy(
        eval_env, 
        arglist["max_steps"], 
        arglist["epochs"], 
        arglist["steps_per_epoch"],
        arglist["sample_model_every"], 
        arglist["update_model_every"],
        rwd_fn=None, 
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
