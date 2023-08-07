# Robust inverse reinforcement learning through Bayesian theory of mind
Bayesian MAP simultaneous estimation of reward and dynamics for offline model-based inverse reinforcement learning. Pytorch implementation of [paper](https://openreview.net/forum?id=iL1rdSiffz).

## Usage
Environment set up:
```
conda env create -f environment.yml
conda activate irl
```

### D4RL MuJoCo experiments
Optional dataset download:
```
python scripts/download_d4rl_data.py
```

Run BTOM IRL:
```
sh scripts/irl/train_btom.sh
```

### Gridworld experiments
Create demonstrations and run IRL (two-stage IRL as default, change settings in the .sh script):
```
sh scripts/tabular/create_gridworld_demonstrations.sh
sh scripts/tabular/train_gridworld.sh
```

## Implemented algorithms
See ```src/agents``` and ```src/algo``` for additional implemented RL and IRL algorithms. 