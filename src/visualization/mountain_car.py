import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MountaincarVis:
    def __init__(self, env):
        self.env = env
    
    def plot_value_map(self, v, ax, fmt=".2f", annot=False, cbar=True, cmap=None):
        v_map = v.reshape(self.env.v_bins, self.env.x_bins).T
        df_v_map = pd.DataFrame(v_map, columns=self.env.x_grid.round(2), index=self.env.v_grid.round(2))
        sns.heatmap(df_v_map, fmt=fmt, annot=annot, cbar=cbar, cmap=cmap, ax=ax)
        ax.invert_yaxis()
        return ax

    def plot_sample_path(self, s_seq, ax):
        """
        Args:
            s_seq (np.array): batch of state sequences. size=[batch_size, T]
        """
        sample_path = np.stack([self.env.state2obs(d) for d in s_seq]).astype(float)

        ax.plot(sample_path[:, :, 0].T, sample_path[:, :, 1].T, "k-")
        ax.plot(sample_path[:, 0, 0], sample_path[:, 0, 1], "ro", label="Start")
        ax.plot(sample_path[:, -1, 0], sample_path[:, -1, 1], "go", label="End")
        ax.legend()
        return ax

if __name__ == "__main__":
    import os
    import pickle
    from src.env.mountain_car import DiscreteMountainCar
    np.random.seed(0)

    x_grids = 20
    v_grids = 20
    env = DiscreteMountainCar(x_grids, v_grids)
    vis = MountaincarVis(env)

    data_path = "../../data/mountain_car"
    with open(os.path.join(data_path, "data.p"), "rb") as f:
        data = pickle.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    vis.plot_value_map(env.reward, ax)
    plt.tight_layout()
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    vis.plot_sample_path(data["s"], ax)
    plt.tight_layout()

    plt.show()

    



