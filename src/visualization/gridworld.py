import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridworldVis:
    def __init__(self, env):
        self.env = env

    def value2map(self, v):
        num_grids = self.env.num_grids

        v_map = np.zeros((num_grids, num_grids))
        for i in range(num_grids): # x pos
            for j in range(num_grids): # y pos
                v_map[j, i] = v[self.env.pos2state(np.array([i, j]))]
        return v_map
    
    def plot_value_map(self, v, ax, annot=True, cbar=False, cmap=None, **kwargs):
        v_map = self.value2map(v)
        sns.heatmap(v_map, fmt=".2f", annot=annot, cbar=cbar, cmap=cmap, linewidths=1, linecolor="black", ax=ax, **kwargs)
        ax.invert_yaxis()
        return ax

    def plot_sample_path(self, s_seq, ax, eps=0.1):
        """
        Args:
            s_seq (np.array): batch of state sequences. size=[batch_size, T]
        """
        sample_path = np.stack([self.env.state2pos[d] for d in s_seq]).astype(float)
        sample_path += np.random.normal(size=sample_path.shape) * eps

        ax.plot(sample_path[:, :, 0].T, sample_path[:, :, 1].T, "k-")
        ax.plot(sample_path[:, 0, 0], sample_path[:, 0, 1], "ro", label="Start")
        ax.plot(sample_path[:, -1, 0], sample_path[:, -1, 1], "go", label="End")
        ax.legend()
        return ax
    
    def plot_state_action(self, p, ax, fmt=".2f", annot=True, cbar=False, vmin=None, vmax=None):
        sns.heatmap(p, fmt=fmt, vmin=vmin, vmax=vmax, annot=annot, cbar=cbar, ax=ax)
        s_labels = self.env.state2pos
        ax.set_xticklabels(self.env.a_labels)
        ax.set_yticklabels(
            [tuple(s_labels[int(l - 0.5)]) for l in ax.get_yticks()],
            rotation=0
        )
        ax.invert_yaxis()
        return ax

if __name__ == "__main__":
    import os
    import pickle
    from src.env.gridworld import Gridworld
    np.random.seed(0)

    num_grids = 5
    env = Gridworld(num_grids)
    vis = GridworldVis(env)

    data_path = "../../data/gridworld"
    with open(os.path.join(data_path, "data_one_state_one_goal.p"), "rb") as f:
        data = pickle.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    vis.plot_value_map(env.init_dist, ax)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    vis.plot_sample_path(data["s"], ax)
    plt.show()

    



