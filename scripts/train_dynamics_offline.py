import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch

from src.agents.mopo import MBPO
from src.agents.rl_utils import Logger
from src.algo.logging_utils import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    list_ = lambda x: [float(i.replace(" ", "")) for i in x.split(",")]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp/dynamics")
    parser.add_argument("--data_path", type=str, default="../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # algo args
    parser.add_argument("--ensemble_dim", type=int, default=5, help="ensemble size, default=5")
    parser.add_argument("--hidden_dim", type=int, default=200, help="neural network hidden dims, default=128")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--clip_lv", type=bool_, default=False, help="whether to clip observation variance, default=False")
    # data args
    parser.add_argument("--num_samples", type=int, default=100000, help="number of training transitions, default=100000")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="train test split ratio, default=0.2")
    # training args
    parser.add_argument("--batch_size", type=int, default=200, help="training batch size, default=200")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default=0.001")
    parser.add_argument("--decay", type=list_, default=[0.000025, 0.00005, 0.000075, 0.0001], 
        help="weight decay for each layer, default=[0.000025, 0.00005, 0.000075, 0.0001]")
    parser.add_argument("--grad_clip", type=float, default=100., help="gradient clipping, default=100.")
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of reward training epochs, default=10")
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
    
    # init model
    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]
    act_lim = torch.ones(act_dim)
    agent = MBPO(
        obs_dim, 
        act_dim, 
        act_lim, 
        arglist["ensemble_dim"], 
        arglist["hidden_dim"], 
        arglist["num_hidden"], 
        arglist["activation"],
        clip_lv=arglist["clip_lv"], 
        eval_ratio=arglist["eval_ratio"],
        lr_m=arglist["lr"], 
        decay=arglist["decay"], 
        grad_clip=arglist["grad_clip"]
    )
    plot_keys = ["obs_loss", "obs_mae", "rwd_loss", "rwd_mae"]
    
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
    
    agent.real_buffer.push_batch(
        obs, act, rwd, next_obs, terminated
    )
    agent.update_stats() # update stats to normalize data in agent
    print(agent)
    
    # init save callback
    callback = None
    if arglist["save"]:
        callback = SaveCallback(arglist, plot_keys, cp_history=cp_history)
    
    # training loop
    logger = Logger()
    best_eval = 1e6
    epoch_since_last_update = 0
    for e in range(arglist["epochs"]):
        agent.train_model_epoch(1, logger)
        logger.push({"epoch": e + 1})
        logger.log()
        print()

        if callback is not None:
            callback(agent, logger)
        
        # termination condition based on eval performance
        current_eval = logger.history[-1]["rwd_mae"] + logger.history[-1]["obs_mae"]
        improvement = (best_eval - current_eval) / best_eval
        if improvement > 0.01:
            epoch_since_last_update = 0
        else:
            epoch_since_last_update += 1
        best_eval = min(best_eval, current_eval)
        
        if epoch_since_last_update > 5:
            break

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.save_history(pd.DataFrame(logger.history))


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
