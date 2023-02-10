import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch

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
        if plot_std and plot_keys[i].replace("_avg", "_std") in df_history.columns:
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
        exp_path = os.path.join(arglist["exp_path"])
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
        optimizer_state_dict = {
            optimizer_name: {k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in optimizer.state_dict().items()}
            for optimizer_name, optimizer in model.optimizers.items()
        }
        
        torch.save({
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }, path)
        print(f"\ncheckpoint saved at: {path}\n")