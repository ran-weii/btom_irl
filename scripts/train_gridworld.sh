#! /bin/bash
num_traj=300
exact=True
save=True

# btom settings
algo=btom
fit_transition=True
fit_reward=True
mle_transition=False
pess_penalty=0.
lr=0.05

# irl settings
# algo=irl
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=0.
# lr=0.1

# pil settings
# algo=pil
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=1.
# lr=0.1

python train_gridworld.py \
--init_type one_state \
--goal_type one_goal \
--p_goal 0.95 \
--epsilon 0. \
--num_traj $num_traj \
--gamma 0.9 \
--alpha 1. \
--horizon 0 \
--algo $algo \
--fit_transition $fit_transition \
--fit_reward $fit_reward \
--mle_transition $mle_transition \
--pess_penalty $pess_penalty \
--rollout_steps 100 \
--exact $exact \
--obs_penalty 10. \
--lr $lr \
--decay 0. \
--epochs 1000 \
--save $save

python train_gridworld.py \
--init_type one_state \
--goal_type three_goals \
--p_goal 0.95 \
--epsilon 0. \
--num_traj $num_traj \
--gamma 0.9 \
--alpha 1. \
--horizon 0 \
--algo $algo \
--fit_transition $fit_transition \
--fit_reward $fit_reward \
--mle_transition $mle_transition \
--pess_penalty $pess_penalty \
--rollout_steps 100 \
--exact $exact \
--obs_penalty 10. \
--lr $lr \
--decay 0. \
--epochs 1000 \
--save $save

python train_gridworld.py \
--init_type uniform \
--goal_type three_goals \
--p_goal 0.95 \
--epsilon 0. \
--num_traj $num_traj \
--gamma 0.9 \
--alpha 1. \
--horizon 0 \
--algo $algo \
--fit_transition $fit_transition \
--fit_reward $fit_reward \
--mle_transition $mle_transition \
--pess_penalty $pess_penalty \
--rollout_steps 100 \
--exact $exact \
--obs_penalty 10. \
--lr $lr \
--decay 0. \
--epochs 1000 \
--save $save