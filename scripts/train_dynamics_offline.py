import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.algo.offline_irl import OfflineIRL
from src.algo.logging_utils import SaveCallback

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp/dynamics")
    parser.add_argument("--data_path", type=str, default="../data/d4rl/")
    parser.add_argument("--filename", type=str, default="hopper-expert-v2.p")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # algo args
    parser.add_argument("--ensemble_dim", type=int, default=5, help="ensemble size, default=5")
    parser.add_argument("--hidden_dim", type=int, default=128, help="neural network hidden dims, default=128")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--clip_lv", type=bool_, default=False, help="whether to clip observation variance, default=False")
    # data args
    parser.add_argument("--num_samples", type=int, default=100000, help="number of training transitions, default=100000")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train test split ratio, default=0.8")
    # training args
    parser.add_argument("--batch_size", type=int, default=200, help="training batch size, default=200")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=1e-5")
    parser.add_argument("--grad_clip", type=float, default=100., help="gradient clipping, default=100.")
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of reward training epochs, default=10")
    parser.add_argument("--cp_every", type=int, default=10, help="checkpoint interval, default=10")
    parser.add_argument("--verbose", type=bool_, default=True)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

class CustomDataset(Dataset):
    def __init__(self, obs, act, next_obs):
        super().__init__()
        self.obs = torch.from_numpy(obs).to(torch.float32)
        self.act = torch.from_numpy(act).to(torch.float32)
        self.next_obs = torch.from_numpy(next_obs).to(torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {"obs": self.obs[idx], "act": self.act[idx], "next_obs": self.next_obs[idx]}

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
    next_obs = dataset["next_observations"]
    
    # subsample data
    num_samples = arglist["num_samples"]
    train_ratio = arglist["train_ratio"]
    num_train = int(num_samples * train_ratio)

    idx = np.arange(len(obs))
    np.random.shuffle(idx)

    idx_train = idx[:num_train]
    idx_test = idx[num_train:num_samples]

    obs_train = obs[idx_train]
    act_train = act[idx_train]
    next_obs_train = next_obs[idx_train]

    obs_test = obs[idx_test]
    act_test = act[idx_test]
    next_obs_test = next_obs[idx_test]

    # normalize data
    obs_mean = obs_train.mean(0)
    obs_std = obs_train.std(0)    

    obs_train_norm = (obs_train - obs_mean) / obs_std
    next_obs_train_norm = (next_obs_train - obs_mean) / obs_std

    obs_test_norm = (obs_test - obs_mean) / obs_std
    next_obs_test_norm = (next_obs_test - obs_mean) / obs_std
    
    print("train data size:", obs_train.shape, act_train.shape, next_obs_train.shape)
    print("test data size:", obs_test.shape, act_test.shape, next_obs_test.shape)
    
    train_set = CustomDataset(obs_train_norm, act_train, next_obs_train_norm)
    test_set = CustomDataset(obs_test_norm, act_test, next_obs_test_norm)
    
    train_loader = DataLoader(train_set, arglist["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, arglist["batch_size"], shuffle=False)

    # init model
    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]
    act_lim = torch.ones(act_dim)
    agent = OfflineIRL(
        obs_dim, act_dim, act_lim, 
        arglist["ensemble_dim"], arglist["hidden_dim"], arglist["num_hidden"], arglist["activation"],
        clip_lv=arglist["clip_lv"], lr_a=arglist["lr"], decay=arglist["decay"], grad_clip=arglist["grad_clip"]
    )
    plot_keys = ["obs_loss_avg", "obs_mae_avg"]
    
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
    logger = agent.train_model_offline(train_loader, test_loader, arglist["epochs"], callback=callback)

    if arglist["save"]:
        callback.save_checkpoint(agent)
        callback.save_history(pd.DataFrame(logger.history))

        # save normalization stats
        with open(os.path.join(callback.save_path, "norm_stats.npy"), "wb") as f:
            np.save(f, obs_mean)
            np.save(f, obs_std)


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
