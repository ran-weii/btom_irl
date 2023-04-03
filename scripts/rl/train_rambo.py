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

from src.utils.data import load_data
from src.agents.dynamics import EnsembleDynamics, train_ensemble
from src.agents.rambo import RAMBO
from src.env.gym_wrapper import GymEnv, get_termination_fn
from src.utils.logging import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="rambo")
    parser.add_argument("--exp_path", type=str, default="../../exp/mujoco/rl")
    parser.add_argument("--data_path", type=str, default="../../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--dynamics_path", type=str, default="", 
        help="pretrained dynamics path, default=none")
    # data args
    parser.add_argument("--num_samples", type=int, default=2000000, help="number of training transitions, default=2000000")
    parser.add_argument("--norm_obs", type=bool_, default=True, help="normalize observatins, default=True")
    parser.add_argument("--norm_rwd", type=bool_, default=False, help="normalize reward, default=False")
    # dynamics args
    parser.add_argument("--ensemble_dim", type=int, default=7, help="ensemble size, default=7")
    parser.add_argument("--topk", type=int, default=5, help="top k models to perform rollout, default=5")
    parser.add_argument("--m_hidden_dim", type=int, default=200, help="dynamics neural network hidden dims, default=200")
    parser.add_argument("--m_num_hidden", type=int, default=3, help="dynamics number of hidden layers, default=3")
    parser.add_argument("--m_activation", type=str, default="silu", help="dynamics neural network activation, default=silu")
    parser.add_argument("--clip_lv", type=bool_, default=True, help="whether to clip observation variance, default=True")
    parser.add_argument("--residual", type=bool_, default=True, help="whether to predict observation residual, default=True")
    parser.add_argument("--min_std", type=float, default=0.04, help="dynamics minimum prediction std, default=0.04")
    parser.add_argument("--max_std", type=float, default=1.6, help="dynamics maximum prediction std, default=1.6")
    parser.add_argument("--obs_penalty", type=float, default=1., help="transition likelihood penalty, default=1.")
    parser.add_argument("--adv_penalty", type=float, default=0.08, help="model advantage penalty, default=0.08")
    parser.add_argument("--adv_clip_max", type=float, default=60., help="clip advantage max value, default=60.")
    parser.add_argument("--adv_action_deterministic", type=bool_, default=True, help="whether to use deterministic action in advantage, default=True")
    parser.add_argument("--adv_include_entropy", type=bool_, default=False, help="whether to include entropy in advantage, default=False")
    parser.add_argument("--norm_advantage", type=bool_, default=True, help="whether to normalize advantage, default=True")
    parser.add_argument("--update_critic_adv", type=bool_, default=False, help="whether to update critic during model training, default=False")
    parser.add_argument("--decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001]")
    # policy args
    parser.add_argument("--a_hidden_dim", type=int, default=256, help="policy neural network hidden dims, default=256")
    parser.add_argument("--a_num_hidden", type=int, default=2, help="policy number of hidden layers, default=2")
    parser.add_argument("--a_activation", type=str, default="relu", help="policy neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.99, help="trainer discount factor, default=0.99")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--tune_beta", type=bool_, default=True, help="whether to tune beta, default=True")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e6, help="replay buffer size, default=1e6")
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size, default=256")
    parser.add_argument("--rollout_batch_size", type=int, default=50000, help="model rollout batch size, default=50000")
    parser.add_argument("--rollout_deterministic", type=bool_, default=False, help="whether to rollout deterministically, default=False")
    parser.add_argument("--rollout_min_steps", type=int, default=5, help="min dynamics rollout steps, default=5")
    parser.add_argument("--rollout_max_steps", type=int, default=5, help="max dynamics rollout steps, default=5")
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
    parser.add_argument("--seed", type=int, default=0)
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
    filepath = os.path.join(arglist["data_path"], arglist["filename"])
    obs, act, rwd, next_obs, terminated = load_data(filepath, arglist["num_samples"])
    
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
    print("data size:", len(obs))
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
    
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        pred_rwd=True,
        ensemble_dim=arglist["ensemble_dim"],
        topk=arglist["topk"],
        hidden_dim=arglist["m_hidden_dim"],
        num_hidden=arglist["m_num_hidden"],
        activation=arglist["m_activation"],
        decay=arglist["decay"],
        clip_lv=arglist["clip_lv"],
        residual=arglist["residual"],
        termination_fn=termination_fn,
        min_std=arglist["min_std"],
        max_std=arglist["max_std"],
        device=device
    )
    agent = RAMBO(
        dynamics,
        obs_dim, 
        act_dim, 
        act_lim, 
        arglist["a_hidden_dim"], 
        arglist["a_num_hidden"], 
        arglist["a_activation"],
        gamma=arglist["gamma"], 
        beta=arglist["beta"], 
        polyak=arglist["polyak"],
        tune_beta=arglist["tune_beta"],
        obs_penalty=arglist["obs_penalty"], 
        adv_penalty=arglist["adv_penalty"], 
        adv_clip_max=arglist["adv_clip_max"],
        adv_include_entropy=arglist["adv_include_entropy"],
        norm_advantage=arglist["norm_advantage"],
        update_critic_adv=arglist["update_critic_adv"],
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
    
    print("agent norm stats")
    print(agent.dynamics.obs_mean.data.numpy().round(2))
    print((agent.dynamics.obs_variance**0.5).data.numpy().round(2))
    print(agent.dynamics.rwd_mean.data.numpy().round(2))
    print((agent.dynamics.rwd_variance**0.5).data.numpy().round(2))
    
    # init save callback
    callback = None
    if arglist["save"]:
        save_path = os.path.join(arglist["exp_path"], arglist["env_name"], arglist["algo"])
        callback = SaveCallback(arglist, save_path, plot_keys, cp_history)
    
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
    dynamics_pretrain_optimizer = torch.optim.Adam(agent.dynamics.parameters(), lr=arglist["lr_m"])
    train_ensemble(
        [obs, act, rwd, next_obs],
        agent,
        optimizer=dynamics_pretrain_optimizer,
        eval_ratio=arglist["eval_ratio"],
        batch_size=arglist["batch_size"],
        epochs=arglist["pretrain_steps"],
        grad_clip=arglist["grad_clip"],
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
        callback.save_history(pd.DataFrame(logger.history))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
